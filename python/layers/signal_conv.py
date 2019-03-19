# -*- coding: utf-8 -*-
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
"""Signal processing convolution layers.

An alternative abstraction layer for convolution operators that feels more
signal-processingy. Mostly, it has different padding, down-/upsampling, and
alignment handling than `tf.layers.Conv?D`.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

# Dependency imports

from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.layers import base
from tensorflow.python.layers import utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import nn

from tensorflow_compression.python.layers import parameterizers
from tensorflow_compression.python.ops import padding_ops


class _SignalConv(base.Layer):
  """{rank}D convolution layer.

  This layer creates a filter kernel that is convolved or cross correlated with
  the layer input to produce an output tensor. The main difference of this class
  to `tf.layers.Conv?D` is how padding, up- and downsampling, and alignment is
  handled.

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
  will throw an exception.

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
    kernel_initializer: An initializer for the filter kernel.
    bias_initializer: An initializer for the bias vector.
    kernel_regularizer: Optional regularizer for the filter kernel.
    bias_regularizer: Optional regularizer for the bias vector.
    activity_regularizer: Regularizer function for the output.
    kernel_parameterizer: Reparameterization applied to filter kernel. If not
      `None`, must be a `Parameterizer` object. Defaults to `RDFTParameterizer`.
    bias_parameterizer: Reparameterization applied to bias. If not `None`, must
      be a `Parameterizer` object.
    trainable: Boolean. Whether the layer should be trained.
    name: String. The name of the layer.
    dtype: Default dtype of the layer's parameters (default of `None` means use
      the type of the first input).

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
      inputs, i.e. after any reparameterizations.
    bias: `Tensor`-like object. The bias vector as applied to the inputs, i.e.
      after any reparameterizations.
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

  def __init__(self, rank, filters, kernel_support,
               corr=False, strides_down=1, strides_up=1, padding="valid",
               extra_pad_end=True, channel_separable=False,
               data_format="channels_last",
               activation=None, use_bias=False,
               kernel_initializer=init_ops.VarianceScaling(),
               bias_initializer=init_ops.Zeros(),
               kernel_regularizer=None, bias_regularizer=None,
               kernel_parameterizer=parameterizers.RDFTParameterizer(),
               bias_parameterizer=None,
               **kwargs):
    super(_SignalConv, self).__init__(**kwargs)
    self._rank = int(rank)
    self._filters = int(filters)
    self._kernel_support = utils.normalize_tuple(
        kernel_support, self._rank, "kernel_support")
    self._corr = bool(corr)
    self._strides_down = utils.normalize_tuple(
        strides_down, self._rank, "strides_down")
    self._strides_up = utils.normalize_tuple(
        strides_up, self._rank, "strides_up")
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
    self._data_format = utils.normalize_data_format(data_format)
    self._activation = activation
    self._use_bias = bool(use_bias)
    self._kernel_initializer = kernel_initializer
    self._bias_initializer = bias_initializer
    self._kernel_regularizer = kernel_regularizer
    self._bias_regularizer = bias_regularizer
    self._kernel_parameterizer = kernel_parameterizer
    self._bias_parameterizer = bias_parameterizer
    self.input_spec = base.InputSpec(ndim=self._rank + 2)

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
  def _channel_axis(self):
    return {"channels_first": 1, "channels_last": -1}[self.data_format]

  def _pad_strides(self, strides):
    if self.data_format == "channels_first":
      return (1, 1) + strides
    else:
      return (1,) + strides + (1,)

  def build(self, input_shape):
    input_shape = tensor_shape.TensorShape(input_shape)
    channel_axis = self._channel_axis
    input_channels = input_shape[channel_axis].value
    if input_channels is None:
      raise ValueError("The channel dimension of the inputs must be defined.")
    kernel_shape = self.kernel_support + (input_channels, self.filters)
    if self.channel_separable:
      output_channels = self.filters * input_channels
    else:
      output_channels = self.filters

    if self.kernel_parameterizer is None:
      getter = self.add_variable
    else:
      getter = functools.partial(
          self.kernel_parameterizer, getter=self.add_variable)
    self._kernel = getter(
        name="kernel", shape=kernel_shape, dtype=self.dtype,
        initializer=self.kernel_initializer,
        regularizer=self.kernel_regularizer)

    if self.bias_parameterizer is None:
      getter = self.add_variable
    else:
      getter = functools.partial(
          self.bias_parameterizer, getter=self.add_variable)
    self._bias = None if not self.use_bias else getter(
        name="bias", shape=(output_channels,), dtype=self.dtype,
        initializer=self.bias_initializer, regularizer=self.bias_regularizer)

    self.input_spec = base.InputSpec(
        ndim=self._rank + 2, axes={channel_axis: input_channels})

    super(_SignalConv, self).build(input_shape)

  def call(self, inputs):
    inputs = ops.convert_to_tensor(inputs, dtype=self.dtype)
    input_shape = array_ops.shape(inputs)
    outputs = inputs

    # First, perform any requested padding.
    if self.padding in ("same_zeros", "same_reflect"):
      padding = padding_ops.same_padding_for_kernel(
          self.kernel_support, self.corr, self.strides_up)
      if self.data_format == "channels_last":
        padding = [[0, 0]] + list(padding) + [[0, 0]]
      else:
        padding = [[0, 0], [0, 0]] + list(padding)
      outputs = array_ops.pad(outputs, padding, self._pad_mode)

    # Now, perform valid convolutions/correlations.

    # Not for all possible combinations of (`kernel_support`, `corr`,
    # `strides_up`, `strides_down`) TF ops exist. We implement some additional
    # combinations by manipulating the kernels and toggling `corr`.
    kernel = self.kernel
    corr = self.corr

    # If a convolution with no upsampling is desired, we flip the kernels and
    # use cross correlation to implement it, provided the kernels are odd-length
    # in every dimension (with even-length kernels, the boundary handling
    # would have to change, so we'll throw an error instead).
    if (not corr and
        all(s == 1 for s in self.strides_up) and
        all(s % 2 == 1 for s in self.kernel_support)):
      corr = True
      slices = self._rank * (slice(None, None, -1),) + 2 * (slice(None),)
      kernel = kernel[slices]

    # Similarly, we can implement a cross correlation with no downsampling using
    # convolutions. However, we do this only if upsampling is requested, as we
    # are wasting computation in the boundaries whenever we call the transpose
    # convolution ops.
    if (corr and
        all(s == 1 for s in self.strides_down) and
        any(s != 1 for s in self.strides_up) and
        all(s % 2 == 1 for s in self.kernel_support)):
      corr = False
      slices = self._rank * (slice(None, None, -1),) + 2 * (slice(None),)
      kernel = kernel[slices]

    data_format = utils.convert_data_format(
        self.data_format, self._rank + 2)
    if (corr and
        self.channel_separable and
        self._rank == 2 and
        all(s == 1 for s in self.strides_up) and
        all(s == self.strides_down[0] for s in self.strides_down)):
      # `nn.depthwise_conv2d_native` performs channel-separable correlations
      # followed by optional downsampling.
      outputs = nn.depthwise_conv2d_native(
          outputs, kernel, strides=self._pad_strides(self.strides_down),
          padding="VALID", data_format=data_format)
    elif (corr and
          all(s == 1 for s in self.strides_up) and
          not self.channel_separable):
      # `nn.convolution` performs correlations followed by optional
      # downsampling.
      outputs = nn.convolution(
          outputs, kernel, strides=self.strides_down, padding="VALID",
          data_format=data_format)
    elif (not corr and
          all(s == 1 for s in self.strides_down) and
          ((not self.channel_separable and 1 <= self._rank <= 3) or
           (self.channel_separable and self.filters == 1 and self._rank == 2 and
            all(s == self.strides_up[0] for s in self.strides_up)))):
      # `nn.conv?d_transpose` perform convolutions, preceded by optional
      # upsampling. Generally, they increase the spatial support of their
      # inputs, so in order to implement 'valid', we need to crop their outputs.

      # Transpose convolutions expect the output and input channels in reversed
      # order. We implement this by swapping those dimensions of the kernel.
      # For channel separable convolutions, we can't currently perform anything
      # other than one filter per channel, so the last dimension needs to be of
      # length one. Since this happens to be the format that the op expects it,
      # we can skip the transpose in that case.
      if not self.channel_separable:
        kernel = array_ops.transpose(
            kernel, list(range(self._rank)) + [self._rank + 1, self._rank])

      # Compute shape of temporary.
      pad_shape = array_ops.shape(outputs)
      temp_shape = [pad_shape[0]] + (self._rank + 1) * [None]
      if self.data_format == "channels_last":
        spatial_axes = range(1, self._rank + 1)
        if self.channel_separable:
          temp_shape[-1] = input_shape[-1]
        else:
          temp_shape[-1] = self.filters
      else:
        spatial_axes = range(2, self._rank + 2)
        if self.channel_separable:
          temp_shape[1] = input_shape[1]
        else:
          temp_shape[1] = self.filters
      if self.extra_pad_end:
        get_length = lambda l, s, k: l * s + (k - 1)
      else:
        get_length = lambda l, s, k: l * s + (k - s)
      for i, a in enumerate(spatial_axes):
        temp_shape[a] = get_length(
            pad_shape[a], self.strides_up[i], self.kernel_support[i])

      # Compute convolution.
      if self._rank == 1 and not self.channel_separable:
        # There's no 1D transpose convolution op, so we insert an extra
        # dimension and use 2D.
        extradim = {"channels_first": 2, "channels_last": 1}[self.data_format]
        strides = self._pad_strides(self.strides_up)
        temp = array_ops.squeeze(
            nn.conv2d_transpose(
                array_ops.expand_dims(outputs, extradim),
                array_ops.expand_dims(kernel, 0),
                temp_shape[:extradim] + [1] + temp_shape[extradim:],
                strides=strides[:extradim] + (1,) + strides[extradim:],
                padding="VALID", data_format=data_format.replace("W", "HW")),
            [extradim])
      elif self._rank == 2 and self.channel_separable:
        temp = nn.depthwise_conv2d_native_backprop_input(
            temp_shape, kernel, outputs,
            strides=self._pad_strides(self.strides_up), padding="VALID",
            data_format=data_format)
      elif self._rank == 2 and not self.channel_separable:
        temp = nn.conv2d_transpose(
            outputs, kernel, temp_shape,
            strides=self._pad_strides(self.strides_up), padding="VALID",
            data_format=data_format)
      elif self._rank == 3 and not self.channel_separable:
        temp = nn.conv3d_transpose(
            outputs, kernel, temp_shape,
            strides=self._pad_strides(self.strides_up), padding="VALID",
            data_format=data_format)
      else:
        assert False  # Should never reach this.

      # Perform crop.
      slices = [slice(None)] * (self._rank + 2)
      if self.padding == "valid":
        # Take `kernel_support - 1` samples away from both sides. This leaves
        # just samples computed without padding.
        for i, a in enumerate(spatial_axes):
          slices[a] = slice(
              self.kernel_support[i] - 1,
              None if self.kernel_support[i] == 1 else
              1 - self.kernel_support[i])
      else:
        # Take `kernel_support // 2` plus the padding away from beginning, and
        # crop end to input length multiplied by upsampling factor.
        for i, a in enumerate(spatial_axes):
          offset = padding[a][0] * self.strides_up[i]
          offset += self.kernel_support[i] // 2
          length = get_length(input_shape[a], self.strides_up[i], offset + 1)
          slices[a] = slice(offset, length)
      outputs = temp[slices]
    else:
      raise NotImplementedError(
          "The provided combination of SignalConv arguments is not currently "
          "implemented (kernel_support={}, corr={}, strides_down={}, "
          "strides_up={}, channel_separable={}, filters={}). "
          "Try using odd-length kernels or turning off separability?".format(
              self.kernel_support, self.corr, self.strides_down,
              self.strides_up, self.channel_separable, self.filters))

    # Now, add bias if requested.
    if self.bias is not None:
      if self.data_format == "channels_first":
        # As of Mar 2017, direct addition is significantly slower than
        # bias_add when computing gradients.
        if self._rank == 1:
          # nn.bias_add does not accept a 1D input tensor.
          outputs = array_ops.expand_dims(outputs, 2)
          outputs = nn.bias_add(outputs, self.bias, data_format="NCHW")
          outputs = array_ops.squeeze(outputs, [2])
        elif self._rank == 2:
          outputs = nn.bias_add(outputs, self.bias, data_format="NCHW")
        elif self._rank >= 3:
          shape = array_ops.shape(outputs)
          outputs = array_ops.reshape(outputs, shape[:3] + [-1])
          outputs = nn.bias_add(outputs, self.bias, data_format="NCHW")
          outputs = array_ops.reshape(outputs, shape)
      else:
        outputs = nn.bias_add(outputs, self.bias)

    # Finally, pass through activation function if requested.
    if self.activation is not None:
      outputs = self.activation(outputs)  # pylint:disable=not-callable

    # Aid shape inference, for some reason shape info is not always available.
    if not context.executing_eagerly():
      outputs.set_shape(self.compute_output_shape(inputs.shape))

    return outputs

  def compute_output_shape(self, input_shape):
    input_shape = tensor_shape.TensorShape(input_shape)
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
      return tensor_shape.TensorShape([batch, channels] + spatial)
    else:
      return tensor_shape.TensorShape([batch] + spatial + [channels])


def _conv_class_factory(name, rank):
  """Subclass from _SignalConv, fixing convolution rank."""
  def init(self, *args, **kwargs):
    return _SignalConv.__init__(self, rank, *args, **kwargs)
  clsdict = {"__init__": init,
             "__doc__": _SignalConv.__doc__.format(rank=rank)}
  return type(name, (_SignalConv,), clsdict)


# pylint:disable=invalid-name
# Subclass _SignalConv for each dimensionality.
SignalConv1D = _conv_class_factory("SignalConv1D", 1)
SignalConv2D = _conv_class_factory("SignalConv2D", 2)
SignalConv3D = _conv_class_factory("SignalConv3D", 3)
# pylint:enable=invalid-name
