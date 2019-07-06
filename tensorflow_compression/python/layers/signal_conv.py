# Copyright 2018 Google LLC. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Swiss army tool for convolutions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

import tensorflow as tf

from tensorflow_compression.python.layers import parameterizers
from tensorflow_compression.python.ops import padding_ops


__all__ = [
    "SignalConv1D",
    "SignalConv2D",
    "SignalConv3D",
]


class _SignalConv(tf.keras.layers.Layer):
  """{rank}D convolution layer.

  This layer creates a filter kernel that is convolved or cross correlated with
  the layer input to produce an output tensor. The main difference of this class
  to `tf.layers.Conv{rank}D` is how padding, up- and downsampling, and alignment
  is handled. It supports much more flexible options for structuring the linear
  transform.

  In general, the outputs are equivalent to a composition of:
  1. an upsampling step (if `strides_up > 1`)
  2. a convolution or cross correlation
  3. a downsampling step (if `strides_down > 1`)
  4. addition of a bias vector (if `use_bias == True`)
  5. a pointwise nonlinearity (if `activation is not None`)

  For more information on what the difference between convolution and cross
  correlation is, see [this](https://en.wikipedia.org/wiki/Convolution) and
  [this](https://en.wikipedia.org/wiki/Cross-correlation) Wikipedia article,
  respectively. Note that the distinction between convolution and cross
  correlation is occasionally blurred (one may use convolution as an umbrella
  term for both). For a discussion of up-/downsampling, refer to the articles
  about [upsampling](https://en.wikipedia.org/wiki/Upsampling) and
  [decimation](https://en.wikipedia.org/wiki/Decimation_(signal_processing)). A
  more in-depth treatment of all of these operations can be found in:

  > "Discrete-Time Signal Processing"<br />
  > Oppenheim, Schafer, Buck (Prentice Hall)

  For purposes of this class, the center position of a kernel is always
  considered to be at `K // 2`, where `K` is the support length of the kernel.
  This implies that in the `'same_*'` padding modes, all of the following
  operations will produce the same result if applied to the same inputs, which
  is not generally true for convolution operations as implemented by
  `tf.nn.convolution` or `tf.layers.Conv?D` (numbers represent kernel
  coefficient values):

  - convolve with `[1, 2, 3]`
  - convolve with `[0, 1, 2, 3, 0]`
  - convolve with `[0, 1, 2, 3]`
  - correlate with `[3, 2, 1]`
  - correlate with `[0, 3, 2, 1, 0]`
  - correlate with `[0, 3, 2, 1]`

  Available padding (boundary handling) modes:

  - `'valid'`: This always yields the maximum number of output samples that can
    be computed without making any assumptions about the values outside of the
    support of the input tensor. The padding semantics are always applied to the
    inputs. In contrast, even though `tf.nn.conv2d_transpose` implements
    upsampling, in `'VALID'` mode it will produce an output tensor with *larger*
    support than the input tensor (because it is the transpose of a `'VALID'`
    downsampled convolution).

    Examples (numbers represent indexes into the respective tensors, periods
    represent skipped spatial positions):

    `kernel_support = 5` and `strides_down = 2`:
    ```
    inputs:  |0 1 2 3 4 5 6 7 8|
    outputs: |    0 . 1 . 2    |
    ```
    ```
    inputs:  |0 1 2 3 4 5 6 7|
    outputs: |    0 . 1 .    |
    ```

    `kernel_support = 3`, `strides_up = 2`, and `extra_pad_end = True`:
    ```
    inputs:   |0 . 1 . 2 . 3 . 4 .|
    outputs:  |  0 1 2 3 4 5 6 7  |
    ```

    `kernel_support = 3`, `strides_up = 2`, and `extra_pad_end = False`:
    ```
    inputs:   |0 . 1 . 2 . 3 . 4|
    outputs:  |  0 1 2 3 4 5 6  |
    ```

  - `'same_zeros'`: Values outside of the input tensor support are assumed to be
    zero. Similar to `'SAME'` in `tf.nn.convolution`, but with different
    padding. In `'SAME'`, the spatial alignment of the output depends on the
    input shape. Here, the output alignment depends only on the kernel support
    and the strides, making alignment more predictable. The first sample in the
    output is always spatially aligned with the first sample in the input.

    Examples (numbers represent indexes into the respective tensors, periods
    represent skipped spatial positions):

    `kernel_support = 5` and `strides_down = 2`:
    ```
    inputs:  |0 1 2 3 4 5 6 7 8|
    outputs: |0 . 1 . 2 . 3 . 4|
    ```
    ```
    inputs:  |0 1 2 3 4 5 6 7|
    outputs: |0 . 1 . 2 . 3 .|
    ```

    `kernel_support = 3`, `strides_up = 2`, and `extra_pad_end = True`:
    ```
    inputs:   |0 . 1 . 2 . 3 . 4 .|
    outputs:  |0 1 2 3 4 5 6 7 8 9|
    ```

    `kernel_support = 3`, `strides_up = 2`, and `extra_pad_end = False`:
    ```
    inputs:   |0 . 1 . 2 . 3 . 4|
    outputs:  |0 1 2 3 4 5 6 7 8|
    ```

  - `'same_reflect'`: Values outside of the input tensor support are assumed to
    be reflections of the samples inside. Note that this is the same padding as
    implemented by `tf.pad` in the `'REFLECT'` mode (i.e. with the symmetry axis
    on the samples rather than between). The output alignment is identical to
    the `'same_zeros'` mode.

    Examples: see `'same_zeros'`.

    When applying several convolutions with down- or upsampling in a sequence,
    it can be helpful to keep the axis of symmetry for the reflections
    consistent. To do this, set `extra_pad_end = False` and make sure that the
    input has length `M`, such that `M % S == 1`, where `S` is the product of
    stride lengths of all subsequent convolutions. Example for subsequent
    downsampling (here, `M = 9`, `S = 4`, and `^` indicate the symmetry axes
    for reflection):

    ```
    inputs:       |0 1 2 3 4 5 6 7 8|
    intermediate: |0 . 1 . 2 . 3 . 4|
    outputs:      |0 . . . 1 . . . 2|
                   ^               ^
    ```

  Note that due to limitations of the underlying operations, not all
  combinations of arguments are currently implemented. In this case, this class
  will throw a `NotImplementedError` exception.

  Speed tips:

  - Prefer combining correlations with downsampling, and convolutions with
    upsampling, as the underlying ops implement these combinations directly.
  - If that isn't desirable, prefer using odd-length kernel supports, since
    odd-length kernels can be flipped if necessary, to use the fastest
    implementation available.
  - Combining upsampling and downsampling (for rational resampling ratios)
    is relatively slow, because no underlying ops exist for that use case.
    Downsampling in this case is implemented by discarding computed output
    values.
  - Note that `channel_separable` is only implemented for 1D and 2D. Also,
    upsampled channel-separable convolutions are currently only implemented for
    `filters == 1`. When using `channel_separable`, prefer using identical
    strides in all dimensions to maximize performance.

  Arguments:
    filters: Integer. If `not channel_separable`, specifies the total number of
      filters, which is equal to the number of output channels. Otherwise,
      specifies the number of filters per channel, which makes the number of
      output channels equal to `filters` times the number of input channels.
    kernel_support: An integer or iterable of {rank} integers, specifying the
      length of the convolution/correlation window in each dimension.
    corr: Boolean. If True, compute cross correlation. If False, convolution.
    strides_down: An integer or iterable of {rank} integers, specifying an
      optional downsampling stride after the convolution/correlation.
    strides_up: An integer or iterable of {rank} integers, specifying an
      optional upsampling stride before the convolution/correlation.
    padding: String. One of the supported padding modes (see above).
    extra_pad_end: Boolean. When upsampling, use extra skipped samples at the
      end of each dimension (default). For examples, refer to the discussion
      of padding modes above.
    channel_separable: Boolean. If `False` (default), each output channel is
      computed by summing over all filtered input channels. If `True`, each
      output channel is computed from only one input channel, and `filters`
      specifies the number of filters per channel. The output channels are
      ordered such that the first block of `filters` channels is computed from
      the first input channel, the second block from the second input channel,
      etc.
    data_format: String, one of `channels_last` (default) or `channels_first`.
      The ordering of the input dimensions. `channels_last` corresponds to
      input tensors with shape `(batch, ..., channels)`, while `channels_first`
      corresponds to input tensors with shape `(batch, channels, ...)`.
    activation: Activation function or `None`.
    use_bias: Boolean, whether an additive constant will be applied to each
      output channel.
    use_explicit: Boolean, whether to use `EXPLICIT` padding mode (supported in
      TensorFlow >1.14).
    kernel_initializer: An initializer for the filter kernel.
    bias_initializer: An initializer for the bias vector.
    kernel_regularizer: Optional regularizer for the filter kernel.
    bias_regularizer: Optional regularizer for the bias vector.
    activity_regularizer: Regularizer function for the output.
    kernel_parameterizer: Reparameterization applied to filter kernel. If not
      `None`, must be a `Parameterizer` object. Defaults to `RDFTParameterizer`.
    bias_parameterizer: Reparameterization applied to bias. If not `None`,
      must be a `Parameterizer` object. Defaults to `None`.
    trainable: Boolean. Whether the layer should be trained.
    name: String. The name of the layer.
    dtype: `DType` of the layer's inputs (default of `None` means use the type
      of the first input).

  Read-only properties:
    filters: See above.
    kernel_support: See above.
    corr: See above.
    strides_down: See above.
    strides_up: See above.
    padding: See above.
    extra_pad_end: See above.
    channel_separable: See above.
    data_format: See above.
    activation: See above.
    use_bias: See above.
    kernel_initializer: See above.
    bias_initializer: See above.
    kernel_regularizer: See above.
    bias_regularizer: See above.
    activity_regularizer: See above.
    kernel_parameterizer: See above.
    bias_parameterizer: See above.
    name: See above.
    dtype: See above.
    kernel: `Tensor`-like object. The convolution kernel as applied to the
      inputs, i.e. after any reparameterizers.
    bias: `Tensor`-like object. The bias vector as applied to the inputs, i.e.
      after any reparameterizers.
    trainable_variables: List of trainable variables.
    non_trainable_variables: List of non-trainable variables.
    variables: List of all variables of this layer, trainable and non-trainable.
    updates: List of update ops of this layer.
    losses: List of losses added by this layer.

  Mutable properties:
    trainable: Boolean. Whether the layer should be trained.
    input_spec: Optional `InputSpec` object specifying the constraints on inputs
      that can be accepted by the layer.
  """

  def __init__(self, filters, kernel_support,
               corr=False, strides_down=1, strides_up=1, padding="valid",
               extra_pad_end=True, channel_separable=False,
               data_format="channels_last",
               activation=None, use_bias=False, use_explicit=False,
               kernel_initializer=tf.initializers.variance_scaling(),
               bias_initializer=tf.initializers.zeros(),
               kernel_regularizer=None, bias_regularizer=None,
               kernel_parameterizer=parameterizers.RDFTParameterizer(),
               bias_parameterizer=None,
               **kwargs):
    super(_SignalConv, self).__init__(**kwargs)

    self._filters = int(filters)
    self._kernel_support = self._normalized_tuple(
        kernel_support, "kernel_support")
    self._corr = bool(corr)
    self._strides_down = self._normalized_tuple(strides_down, "strides_down")
    self._strides_up = self._normalized_tuple(strides_up, "strides_up")
    self._padding = str(padding).lower()
    try:
      self._pad_mode = {
          "valid": None,
          "same_zeros": "CONSTANT",
          "same_reflect": "REFLECT",
      }[self.padding]
    except KeyError:
      raise ValueError("Unsupported padding mode: '{}'".format(padding))
    self._extra_pad_end = bool(extra_pad_end)
    self._channel_separable = bool(channel_separable)
    self._data_format = str(data_format)
    self._activation = activation
    self._use_bias = bool(use_bias)
    self._use_explicit = bool(use_explicit)
    self._kernel_initializer = kernel_initializer
    self._bias_initializer = bias_initializer
    self._kernel_regularizer = kernel_regularizer
    self._bias_regularizer = bias_regularizer
    self._kernel_parameterizer = kernel_parameterizer
    self._bias_parameterizer = bias_parameterizer

    if self.data_format not in ("channels_first", "channels_last"):
      raise ValueError("Unknown data format: '{}'.".format(self.data_format))

    self.input_spec = tf.keras.layers.InputSpec(ndim=self._rank + 2)

  @property
  def filters(self):
    return self._filters

  @property
  def kernel_support(self):
    return self._kernel_support

  @property
  def corr(self):
    return self._corr

  @property
  def strides_down(self):
    return self._strides_down

  @property
  def strides_up(self):
    return self._strides_up

  @property
  def padding(self):
    return self._padding

  @property
  def extra_pad_end(self):
    return self._extra_pad_end

  @property
  def channel_separable(self):
    return self._channel_separable

  @property
  def data_format(self):
    return self._data_format

  @property
  def activation(self):
    return self._activation

  @property
  def use_bias(self):
    return self._use_bias

  @property
  def use_explicit(self):
    return self._use_explicit

  @property
  def kernel_initializer(self):
    return self._kernel_initializer

  @property
  def bias_initializer(self):
    return self._bias_initializer

  @property
  def kernel_regularizer(self):
    return self._kernel_regularizer

  @property
  def bias_regularizer(self):
    return self._bias_regularizer

  @property
  def kernel_parameterizer(self):
    return self._kernel_parameterizer

  @property
  def bias_parameterizer(self):
    return self._bias_parameterizer

  @property
  def kernel(self):
    return self._kernel

  @property
  def bias(self):
    return self._bias

  @property
  def _op_data_format(self):
    fmt = {"channels_first": "NC{}", "channels_last": "N{}C"}[self.data_format]
    return fmt.format({1: "W", 2: "HW", 3: "DHW"}[self._rank])

  def _padded_tuple(self, iterable, fill):
    if self.data_format == "channels_first":
      return (fill, fill) + tuple(iterable)
    else:
      return (fill,) + tuple(iterable) + (fill,)

  def _normalized_tuple(self, value, name):
    try:
      return self._rank * (int(value),)
    except (ValueError, TypeError):
      try:
        value = tuple(int(v) for v in value)
        assert len(value) == self._rank
        return value
      except (ValueError, TypeError, AssertionError):
        raise ValueError(
            "`{}` must be an integer or an iterable of integers with length {}."
            .format(name, self._rank))

  def _greatest_common_factor(self, iterable):
    for f in range(max(iterable), 1, -1):
      if all(i % f == 0 for i in iterable):
        return f
    return 1

  def _raise_notimplemented(self):
    raise NotImplementedError(
        "The provided combination of SignalConv{}D arguments is not currently "
        "implemented (filters={}, kernel_support={}, corr={}, strides_down={}, "
        "strides_up={}, channel_separable={}, data_format={}). Try using "
        "odd-length kernels or turning off separability?".format(
            self._rank, self.filters, self.kernel_support, self.corr,
            self.strides_down, self.strides_up, self.channel_separable,
            self.data_format))

  def build(self, input_shape):
    input_shape = tf.TensorShape(input_shape)
    channel_axis = {"channels_first": 1, "channels_last": -1}[self.data_format]
    input_channels = input_shape[channel_axis].value
    if input_channels is None:
      raise ValueError("The channel dimension of the inputs must be defined.")
    self.input_spec = tf.keras.layers.InputSpec(
        ndim=self._rank + 2, axes={channel_axis: input_channels})

    kernel_shape = self.kernel_support + (input_channels, self.filters)
    if self.channel_separable:
      output_channels = self.filters * input_channels
    else:
      output_channels = self.filters

    kernel_parameterizer = self.kernel_parameterizer
    if kernel_parameterizer is None:
      getter = self.add_variable
    else:
      getter = functools.partial(
          kernel_parameterizer, getter=self.add_variable)
    self._kernel = getter(
        name="kernel", shape=kernel_shape, dtype=self.dtype,
        initializer=self.kernel_initializer,
        regularizer=self.kernel_regularizer)

    if self.use_bias:
      bias_parameterizer = self.bias_parameterizer
      if bias_parameterizer is None:
        getter = self.add_variable
      else:
        getter = functools.partial(
            bias_parameterizer, getter=self.add_variable)
      self._bias = getter(
          name="bias", shape=(output_channels,), dtype=self.dtype,
          initializer=self.bias_initializer, regularizer=self.bias_regularizer)
    else:
      self._bias = None

    super(_SignalConv, self).build(input_shape)

  def _correlate_down_valid(self, inputs, kernel):
    # Computes valid correlation followed by downsampling.

    data_format = self._op_data_format
    strides = self._padded_tuple(self.strides_down, 1)

    if 1 <= self._rank <= 3 and not self.channel_separable:
      # `tf.nn.convolution` computes correlation followed by optional
      # downsampling.
      outputs = tf.nn.convolution(
          inputs, kernel,
          strides=self.strides_down, padding="VALID", data_format=data_format)
    elif self._rank == 1 and self.channel_separable:
      # There is no 1D depthwise correlation op, so if that is requested we
      # insert an extra dimension and use the 2D op.
      extradim = {"channels_first": 2, "channels_last": 1}[self.data_format]
      strides = strides[:extradim] + (strides[extradim],) + strides[extradim:]
      data_format = data_format.replace("W", "HW")
      inputs = tf.expand_dims(inputs, extradim)
      kernel = tf.expand_dims(kernel, 0)
      outputs = tf.nn.depthwise_conv2d_native(
          inputs, kernel,
          strides=strides, padding="VALID", data_format=data_format)
      outputs = tf.squeeze(outputs, [extradim])
    elif self._rank == 2 and self.channel_separable:
      # `tf.nn.depthwise_conv2d_native` performs channel-separable correlations
      # followed by optional downsampling. All strides must be identical. If
      # not, we downsample by the greatest common factor and then downsample
      # the result further.
      gcf = self._greatest_common_factor(self.strides_down)
      strides = self._padded_tuple(self._rank * (gcf,), 1)
      outputs = tf.nn.depthwise_conv2d_native(
          inputs, kernel,
          strides=strides, padding="VALID", data_format=data_format)
      # Perform remaining downsampling.
      slices = tuple(slice(None, None, s // gcf) for s in self.strides_down)
      if any(s.step > 1 for s in slices):
        outputs = outputs[self._padded_tuple(slices, slice(None))]
    else:
      self._raise_notimplemented()

    return outputs

  def _correlate_down_explicit(self, inputs, kernel, padding):
    # Computes correlation followed by downsampling, with arbitrary zero
    # padding.
    data_format = self._op_data_format
    strides = self._padded_tuple(self.strides_down, 1)
    padding = self._padded_tuple(padding, (0, 0))
    do_cast = inputs.dtype.is_integer

    if self._rank == 1 and not self.channel_separable:
      # The 1D correlation op can't do explicit padding, so if that is requested
      # we insert an extra dimension and use the 2D op.
      extradim = {"channels_first": 2, "channels_last": 1}[self.data_format]
      strides = strides[:extradim] + (strides[extradim],) + strides[extradim:]
      padding = padding[:extradim] + ((0, 0),) + padding[extradim:]
      data_format = data_format.replace("W", "HW")
      inputs = tf.expand_dims(inputs, extradim)
      kernel = tf.expand_dims(kernel, 0)
      if do_cast:
        inputs = tf.cast(inputs, tf.float32)
      outputs = tf.nn.conv2d(
          inputs, kernel,
          strides=strides, padding=padding, data_format=data_format)
      if do_cast:
        outputs = tf.cast(tf.math.round(outputs), self.accum_dtype)
      outputs = tf.squeeze(outputs, [extradim])
    elif self._rank == 2 and not self.channel_separable:
      # `tf.nn.conv2d` performs correlations followed by optional downsampling.
      if do_cast:
        inputs = tf.cast(inputs, tf.float32)
      outputs = tf.nn.conv2d(
          inputs, kernel,
          strides=strides, padding=padding, data_format=data_format)
      if do_cast:
        outputs = tf.cast(tf.math.round(outputs), self.accum_dtype)
    else:
      self._raise_notimplemented()

    return outputs

  def _up_convolve_transpose_valid(self, inputs, kernel, prepadding):
    # Computes upsampling followed by convolution, via transpose convolution ops
    # in VALID mode. This is a relatively inefficient implementation of
    # upsampled convolutions, where we need to crop away a lot of the values
    # computed in the boundaries.

    # Transpose convolutions expect the output and input channels in reversed
    # order. We implement this by swapping those dimensions of the kernel.
    # For channel separable convolutions, we can't currently perform anything
    # other than one filter per channel, so the last dimension needs to be of
    # length one. Since this happens to be the format that the op expects it,
    # we can skip the transpose in that case.
    if not self.channel_separable:
      kernel = tf.transpose(
          kernel, list(range(self._rank)) + [self._rank + 1, self._rank])

    # Compute shape of temporary.
    input_shape = tf.shape(inputs)
    temp_shape = [input_shape[0]] + (self._rank + 1) * [None]
    if self.data_format == "channels_last":
      spatial_axes = range(1, self._rank + 1)
      temp_shape[-1] = (
          input_shape[-1] if self.channel_separable else self.filters)
    else:
      spatial_axes = range(2, self._rank + 2)
      temp_shape[1] = input_shape[1] if self.channel_separable else self.filters
    if self.extra_pad_end:
      get_length = lambda l, s, k: l * s + (k - 1)
    else:
      get_length = lambda l, s, k: l * s + ((k - 1) - (s - 1))
    for i, a in enumerate(spatial_axes):
      temp_shape[a] = get_length(
          input_shape[a], self.strides_up[i], self.kernel_support[i])

    data_format = self._op_data_format
    strides = self._padded_tuple(self.strides_up, 1)

    # Compute convolution.
    if self._rank == 1 and not self.channel_separable:
      # There's no 1D equivalent to conv2d_backprop_input, so we insert an
      # extra dimension and use the 2D op.
      extradim = {"channels_first": 2, "channels_last": 1}[self.data_format]
      data_format = data_format.replace("W", "HW")
      strides = strides[:extradim] + (strides[extradim],) + strides[extradim:]
      temp_shape = temp_shape[:extradim] + [1] + temp_shape[extradim:]
      kernel = tf.expand_dims(kernel, 0)
      inputs = tf.expand_dims(inputs, extradim)
      outputs = tf.nn.conv2d_backprop_input(
          temp_shape, kernel, inputs,
          strides=strides, padding="VALID", data_format=data_format)
      outputs = tf.squeeze(outputs, [extradim])
    elif self._rank == 1 and self.channel_separable and self.filters == 1:
      # There's no 1D equivalent to depthwise_conv2d_native_backprop_input, so
      # we insert an extra dimension and use the 2D op.
      extradim = {"channels_first": 2, "channels_last": 1}[self.data_format]
      data_format = data_format.replace("W", "HW")
      strides = strides[:extradim] + (strides[extradim],) + strides[extradim:]
      temp_shape = temp_shape[:extradim] + [1] + temp_shape[extradim:]
      kernel = tf.expand_dims(kernel, 0)
      inputs = tf.expand_dims(inputs, extradim)
      outputs = tf.nn.depthwise_conv2d_native_backprop_input(
          temp_shape, kernel, inputs,
          strides=strides, padding="VALID", data_format=data_format)
      outputs = tf.squeeze(outputs, [extradim])
    elif self._rank == 2 and not self.channel_separable:
      outputs = tf.nn.conv2d_backprop_input(
          temp_shape, kernel, inputs,
          strides=strides, padding="VALID", data_format=data_format)
    elif (self._rank == 2 and self.channel_separable and
          self.filters == 1 and self.strides_up[0] == self.strides_up[1]):
      outputs = tf.nn.depthwise_conv2d_native_backprop_input(
          temp_shape, kernel, inputs,
          strides=strides, padding="VALID", data_format=data_format)
    elif self._rank == 3 and not self.channel_separable:
      outputs = tf.nn.conv3d_transpose(
          inputs, kernel, temp_shape,
          strides=strides, padding="VALID", data_format=data_format)
    else:
      self._raise_notimplemented()

    # Perform crop, taking into account any pre-padding that was applied.
    slices = (self._rank + 2) * [slice(None)]
    for i, a in enumerate(spatial_axes):
      if self.padding == "valid":
        # Take `kernel_support - 1` samples away from both sides. This leaves
        # just samples computed without any padding.
        start = stop = self.kernel_support[i] - 1
      else:  # same
        # Take half of kernel sizes plus the pre-padding away from each side.
        start = prepadding[i][0] * self.strides_up[i]
        start += self.kernel_support[i] // 2
        stop = prepadding[i][1] * self.strides_up[i]
        stop += (self.kernel_support[i] - 1) // 2
      step = self.strides_down[i]
      start = start if start > 0 else None
      stop = -stop if stop > 0 else None
      step = step if step > 1 else None
      slices[a] = slice(start, stop, step)
    if not all(s.start is s.stop is s.step is None for s in slices):
      outputs = outputs[tuple(slices)]

    return outputs

  def _up_convolve_transpose_explicit(self, inputs, kernel, prepadding):
    # Computes upsampling followed by convolution, via transpose convolution ops
    # in EXPLICIT mode. This is an efficient implementation of upsampled
    # convolutions, where we only compute values that are necessary.
    do_cast = inputs.dtype.is_integer

    # conv2d_backprop_input expects the output and input channels in reversed
    # order. We implement this by swapping those dimensions of the kernel.
    kernel = tf.transpose(
        kernel, list(range(self._rank)) + [self._rank + 1, self._rank])

    # Compute explicit padding corresponding to the equivalent conv2d call,
    # and the shape of the output, taking into account any pre-padding.
    input_shape = tf.shape(inputs)
    padding = (self._rank + 2) * [(0, 0)]
    output_shape = [input_shape[0]] + (self._rank + 1) * [None]
    if self.data_format == "channels_last":
      spatial_axes = range(1, self._rank + 1)
      output_shape[-1] = self.filters
    else:
      spatial_axes = range(2, self._rank + 2)
      output_shape[1] = self.filters
    if self.extra_pad_end:
      get_length = lambda l, s, k, p: l * s + ((k - 1) - p)
    else:
      get_length = lambda l, s, k, p: l * s + ((k - 1) - (s - 1) - p)
    for i, a in enumerate(spatial_axes):
      if self.padding == "valid":
        padding[a] = 2 * (self.kernel_support[i] - 1,)
      else:  # same
        padding[a] = (
            prepadding[i][0] * self.strides_up[i] + self.kernel_support[i] // 2,
            prepadding[i][1] * self.strides_up[i] + (
                self.kernel_support[i] - 1) // 2,
        )
      output_shape[a] = get_length(
          input_shape[a], self.strides_up[i], self.kernel_support[i],
          sum(padding[a]))

    data_format = self._op_data_format
    strides = self._padded_tuple(self.strides_up, 1)

    # Compute convolution.
    if self._rank == 1 and not self.channel_separable:
      # There's no 1D equivalent to conv2d_backprop_input, so we insert an
      # extra dimension and use the 2D op.
      extradim = {"channels_first": 2, "channels_last": 1}[self.data_format]
      data_format = data_format.replace("W", "HW")
      strides = strides[:extradim] + (strides[extradim],) + strides[extradim:]
      padding = padding[:extradim] + [(0, 0)] + padding[extradim:]
      output_shape = output_shape[:extradim] + [1] + output_shape[extradim:]
      kernel = tf.expand_dims(kernel, 0)
      inputs = tf.expand_dims(inputs, extradim)
      if do_cast:
        inputs = tf.cast(inputs, tf.float32)
      outputs = tf.nn.conv2d_backprop_input(
          output_shape, kernel, inputs,
          strides=strides, padding=padding, data_format=data_format)
      if do_cast:
        outputs = tf.cast(tf.math.round(outputs), self.accum_dtype)
      outputs = tf.squeeze(outputs, [extradim])
    elif self._rank == 2 and not self.channel_separable:
      if do_cast:
        inputs = tf.cast(inputs, tf.float32)
      outputs = tf.nn.conv2d_backprop_input(
          output_shape, kernel, inputs,
          strides=strides, padding=padding, data_format=data_format)
      if do_cast:
        outputs = tf.cast(tf.math.round(outputs), self.accum_dtype)
    else:
      self._raise_notimplemented()

    # Perform downsampling if it is requested.
    if any(s > 1 for s in self.strides_down):
      slices = tuple(slice(None, None, s) for s in self.strides_down)
      slices = self._padded_tuple(slices, slice(None))
      outputs = outputs[slices]

    return outputs

  def call(self, inputs):
    inputs = tf.convert_to_tensor(inputs)
    outputs = inputs

    # Not for all possible combinations of (`kernel_support`, `corr`,
    # `strides_up`, `strides_down`) TF ops exist. We implement some additional
    # combinations by manipulating the kernels and toggling `corr`.
    kernel = self.kernel
    corr = self.corr

    # If a convolution with no upsampling is desired, we flip the kernels and
    # use cross correlation to implement it, provided the kernels are odd-length
    # in every dimension (with even-length kernels, the boundary handling
    # would have to change).
    if (not corr and
        all(s == 1 for s in self.strides_up) and
        all(s % 2 == 1 for s in self.kernel_support)):
      corr = True
      slices = self._rank * (slice(None, None, -1),) + 2 * (slice(None),)
      kernel = kernel[slices]

    # Similarly, we can implement a cross correlation using convolutions.
    # However, we do this only if upsampling is requested, as we are potentially
    # wasting computation in the boundaries whenever we call the transpose ops.
    elif (corr and
          any(s != 1 for s in self.strides_up) and
          all(s % 2 == 1 for s in self.kernel_support)):
      corr = False
      slices = self._rank * (slice(None, None, -1),) + 2 * (slice(None),)
      kernel = kernel[slices]

    # Compute amount of necessary padding, and determine whether to use built-in
    # padding or to pre-pad with a separate op.
    if self.padding == "valid":
      padding = prepadding = self._rank * ((0, 0),)
    else:  # same_*
      padding = padding_ops.same_padding_for_kernel(
          self.kernel_support, corr, self.strides_up)
      if (self.padding == "same_zeros" and
          not self.channel_separable and
          1 <= self._rank <= 2 and
          self.use_explicit):
        # Don't pre-pad and use built-in EXPLICIT mode.
        prepadding = self._rank * ((0, 0),)
      else:
        # Pre-pad and then use built-in valid padding mode.
        outputs = tf.pad(
            outputs, self._padded_tuple(padding, (0, 0)), self._pad_mode)
        prepadding = padding
        padding = self._rank * ((0, 0),)

    # Compute the convolution/correlation. Prefer EXPLICIT padding ops where
    # possible, but don't use them to implement VALID padding.
    if (corr and
        all(s == 1 for s in self.strides_up) and
        not self.channel_separable and
        1 <= self._rank <= 2 and
        not all(p[0] == p[1] == 0 for p in padding)):
      outputs = self._correlate_down_explicit(outputs, kernel, padding)
    elif (corr and
          all(s == 1 for s in self.strides_up) and
          all(p[0] == p[1] == 0 for p in padding)):
      outputs = self._correlate_down_valid(outputs, kernel)
    elif (not corr and
          not self.channel_separable and
          1 <= self._rank <= 2 and
          self.use_explicit):
      outputs = self._up_convolve_transpose_explicit(
          outputs, kernel, prepadding)
    elif not corr:
      outputs = self._up_convolve_transpose_valid(
          outputs, kernel, prepadding)
    else:
      self._raise_notimplemented()

    # Now, add bias if requested.
    if self.bias is not None:
      bias = self.bias
      if self.data_format == "channels_first":
        # As of Mar 2017, direct addition is significantly slower than
        # bias_add when computing gradients.
        if self._rank == 1:
          # tf.nn.bias_add does not accept a 1D input tensor.
          outputs = tf.expand_dims(outputs, 2)
          outputs = tf.nn.bias_add(outputs, bias, data_format="NCHW")
          outputs = tf.squeeze(outputs, [2])
        elif self._rank == 2:
          outputs = tf.nn.bias_add(outputs, bias, data_format="NCHW")
        elif self._rank >= 3:
          shape = tf.shape(outputs)
          outputs = tf.reshape(
              outputs, tf.concat([shape[:3], [-1]], axis=0))
          outputs = tf.nn.bias_add(outputs, bias, data_format="NCHW")
          outputs = tf.reshape(outputs, shape)
      else:
        outputs = tf.nn.bias_add(outputs, bias)

    # Finally, pass through activation function if requested.
    if self.activation is not None:
      outputs = self.activation(outputs)  # pylint:disable=not-callable

    # Aid shape inference, for some reason shape info is not always available.
    if not tf.executing_eagerly():
      outputs.set_shape(self.compute_output_shape(inputs.shape))

    return outputs

  def compute_output_shape(self, input_shape):
    input_shape = tf.TensorShape(input_shape)
    input_shape = input_shape.with_rank(self._rank + 2)
    batch = input_shape[0]
    if self.data_format == "channels_first":
      spatial = input_shape[2:].dims
      channels = input_shape[1]
    else:
      spatial = input_shape[1:-1].dims
      channels = input_shape[-1]

    for i, s in enumerate(spatial):
      if self.extra_pad_end:
        s *= self.strides_up[i]
      else:
        s = (s - 1) * self.strides_up[i] + 1
      if self.padding == "valid":
        s -= self.kernel_support[i] - 1
      s = (s - 1) // self.strides_down[i] + 1
      spatial[i] = s

    if self.channel_separable:
      channels *= self.filters
    else:
      channels = self.filters

    if self.data_format == "channels_first":
      return tf.TensorShape([batch, channels] + spatial)
    else:
      return tf.TensorShape([batch] + spatial + [channels])


def _conv_class_factory(name, rank):
  """Subclass from _SignalConv, fixing convolution rank."""
  clsdict = {"_rank": rank,
             "__doc__": _SignalConv.__doc__.format(rank=rank)}
  return type(name, (_SignalConv,), clsdict)


# pylint:disable=invalid-name
# Subclass _SignalConv for each dimensionality.
SignalConv1D = _conv_class_factory("SignalConv1D", 1)
SignalConv2D = _conv_class_factory("SignalConv2D", 2)
SignalConv3D = _conv_class_factory("SignalConv3D", 3)
# pylint:enable=invalid-name
