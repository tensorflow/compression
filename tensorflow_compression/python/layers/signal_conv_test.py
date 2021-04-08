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
"""Tests of signal processing convolution layers."""

import os
from absl.testing import parameterized
import numpy as np
import scipy.signal
import tensorflow as tf
from tensorflow_compression.python.layers import initializers
from tensorflow_compression.python.layers import parameters
from tensorflow_compression.python.layers import signal_conv


class SignalConvTest(tf.test.TestCase, parameterized.TestCase):

  def test_invalid_data_format_raises_error(self):
    with self.assertRaises(ValueError):
      signal_conv.SignalConv1D(2, 1, data_format="NHWC")

  def test_variables_are_enumerated(self):
    layer = signal_conv.SignalConv2D(3, 1, use_bias=True)
    layer.build((None, None, None, 2))
    self.assertLen(layer.weights, 3)
    self.assertLen(layer.trainable_weights, 3)
    weight_names = [w.name for w in layer.weights]
    self.assertSameElements(
        weight_names, ["kernel_real:0", "kernel_imag:0", "bias:0"])

  def test_bias_variable_is_not_unnecessarily_created(self):
    layer = signal_conv.SignalConv2D(5, 3, use_bias=False)
    layer.build((None, None, None, 3))
    self.assertLen(layer.weights, 2)
    self.assertLen(layer.trainable_weights, 2)
    weight_names = [w.name for w in layer.weights]
    self.assertSameElements(weight_names, ["kernel_real:0", "kernel_imag:0"])

  def test_variables_are_not_enumerated_when_overridden(self):
    layer = signal_conv.SignalConv2D(1, 1)
    layer.kernel_parameter = [[[[1]]]]
    layer.bias_parameter = [0]
    layer.build((None, None, None, 1))
    self.assertEmpty(layer.weights)
    self.assertEmpty(layer.trainable_weights)

  def test_variables_trainable_state_follows_layer(self):
    layer = signal_conv.SignalConv2D(1, 1, use_bias=True)
    layer.trainable = False
    layer.build((None, None, None, 1))
    self.assertLen(layer.weights, 3)
    self.assertEmpty(layer.trainable_weights)

  def test_attributes_cannot_be_set_after_build(self):
    layer = signal_conv.SignalConv1D(2, 1)
    layer.build((None, None, 2))
    with self.assertRaises(RuntimeError):
      layer.filters = 3
    with self.assertRaises(RuntimeError):
      layer.kernel_support = 3
    with self.assertRaises(RuntimeError):
      layer.corr = True
    with self.assertRaises(RuntimeError):
      layer.strides_down = 2
    with self.assertRaises(RuntimeError):
      layer.strides_up = 2
    with self.assertRaises(RuntimeError):
      layer.padding = "valid"
    with self.assertRaises(RuntimeError):
      layer.extra_pad_end = True
    with self.assertRaises(RuntimeError):
      layer.channel_separable = True
    with self.assertRaises(RuntimeError):
      layer.data_format = "channels_first"
    with self.assertRaises(RuntimeError):
      layer.activation = tf.nn.relu
    with self.assertRaises(RuntimeError):
      layer.use_bias = False
    with self.assertRaises(RuntimeError):
      layer.use_explicit = False
    with self.assertRaises(RuntimeError):
      layer.kernel_parameter = tf.ones((1, 2, 3))
    with self.assertRaises(RuntimeError):
      layer.bias_parameter = tf.ones((3,))
    with self.assertRaises(RuntimeError):
      layer.kernel_initializer = tf.keras.initializers.Ones()
    with self.assertRaises(RuntimeError):
      layer.bias_initializer = tf.keras.initializers.Ones()
    with self.assertRaises(RuntimeError):
      layer.kernel_regularizer = tf.keras.regularizers.L2()
    with self.assertRaises(RuntimeError):
      layer.bias_regularizer = tf.keras.regularizers.L2()

  def test_variables_receive_gradients(self):
    x = tf.random.uniform((1, 5, 2), dtype=tf.float32)
    layer = signal_conv.SignalConv1D(2, 3, use_bias=True)
    with tf.GradientTape() as g:
      y = layer(x)
    grads = g.gradient(y, layer.trainable_weights)
    self.assertLen(grads, 3)
    self.assertNotIn(None, grads)
    grad_shapes = [tuple(g.shape) for g in grads]
    weight_shapes = [tuple(w.shape) for w in layer.trainable_weights]
    self.assertSameElements(grad_shapes, weight_shapes)

  @parameterized.parameters(False, True)
  def test_can_be_saved_within_functional_model(self, build):
    inputs = tf.keras.Input(shape=(None, 2))
    outputs = signal_conv.SignalConv1D(
        1, 3, use_bias=True, activation=tf.nn.relu)(inputs)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    layer = model.get_layer("signal_conv1d")

    with self.subTest(name="layer_created_as_expected"):
      self.assertIsInstance(layer, signal_conv.SignalConv1D)
      self.assertIsInstance(layer.kernel_parameter, parameters.RDFTParameter)
      self.assertIsInstance(layer.bias_parameter, tf.Variable)

    if build:
      x = tf.random.uniform((1, 5, 2), dtype=tf.float32)
      y = model(x)
      weight_names = [w.name for w in model.weights]

    tempdir = self.create_tempdir()
    model_path = os.path.join(tempdir, "model")
    # This should force the model to be reconstructed via configs.
    model.save(model_path, save_traces=False)

    model = tf.keras.models.load_model(model_path)

    layer = model.get_layer("signal_conv1d")
    with self.subTest(name="layer_recreated_as_expected"):
      self.assertIsInstance(layer, signal_conv.SignalConv1D)
      self.assertIsInstance(layer.kernel_parameter, parameters.RDFTParameter)
      self.assertIsInstance(layer.bias_parameter, tf.Variable)

    if build:
      with self.subTest(name="model_outputs_identical"):
        self.assertAllEqual(model(x), y)

      with self.subTest(name="model_weights_identical"):
        self.assertSameElements(weight_names, [w.name for w in model.weights])


class ConvolutionsTest(tf.test.TestCase):
  """Tests SignalConv against scipy.signal."""

  def numpy_upsample(self, inputs, strides_up, extra_pad_end):
    """Upsample a numpy array."""
    input_shape = np.array(inputs.shape, dtype=int)
    strides_up = np.array(strides_up, dtype=int)

    input_shape[2:] *= strides_up
    if not extra_pad_end:
      input_shape[2:] -= strides_up - 1
    inputs_up = np.zeros(input_shape, dtype=np.float32)
    slices = [slice(None), slice(None)]
    slices += [slice(None, None, s) for s in strides_up]
    inputs_up[tuple(slices)] = inputs

    return inputs_up

  def scipy_convolve_valid(self, corr, inputs, kernel, strides_down, strides_up,
                           extra_pad_end, channel_separable):
    """Convolve/correlate using SciPy."""
    convolve = scipy.signal.correlate if corr else scipy.signal.convolve
    slices = tuple(slice(None, None, s) for s in strides_down)

    # Upsample input.
    if not all(s == 1 for s in strides_up):
      inputs = self.numpy_upsample(inputs, strides_up, extra_pad_end)

    channels_out = kernel.shape[-1]
    if channel_separable:
      channels_out *= inputs.shape[1]

    # Get size of output and initialize with zeros.
    outputs = convolve(inputs[0, 0], kernel[..., 0, 0], mode="valid")
    outputs = outputs[slices]
    outputs = np.zeros((1, channels_out) + outputs.shape, dtype=np.float32)

    # Iterate over channels, downsampling outputs as we go.
    for batch in range(inputs.shape[0]):
      for filter_out in range(kernel.shape[-1]):
        for channel_in in range(inputs.shape[1]):
          if channel_separable:
            channel_out = channel_in * kernel.shape[-1] + filter_out
          else:
            channel_out = filter_out
          outputs[batch, channel_out] += convolve(
              inputs[batch, channel_in],
              kernel[..., channel_in, filter_out],
              mode="valid")[slices]

    return outputs

  def run_valid(self, batch, input_support, channels, filters, kernel_support,
                corr, strides_down, strides_up, padding, extra_pad_end,
                channel_separable, data_format, activation, use_bias):
    assert padding == "valid"

    # Create input array.
    inputs = np.random.randint(32, size=(batch, channels) + input_support)
    inputs = inputs.astype(np.float32)
    if data_format != "channels_first":
      tf_inputs = tf.constant(np.moveaxis(inputs, 1, -1))
    else:
      tf_inputs = tf.constant(inputs)

    # Create kernel array.
    kernel = np.random.randint(16, size=kernel_support + (channels, filters))
    kernel = kernel.astype(np.float32)
    tf_kernel = tf.constant(kernel)

    # Run SignalConv* layer.
    layer_class = {
        3: signal_conv.SignalConv1D,
        4: signal_conv.SignalConv2D,
        5: signal_conv.SignalConv3D,
    }[inputs.ndim]
    layer = layer_class(
        filters, kernel_support, corr=corr, strides_down=strides_down,
        strides_up=strides_up, padding="valid", extra_pad_end=extra_pad_end,
        channel_separable=channel_separable, data_format=data_format,
        activation=activation, use_bias=use_bias, kernel_parameter=tf_kernel)
    tf_outputs = layer(tf_inputs)
    outputs = tf_outputs.numpy()

    # Check that SignalConv* computes the correct output size.
    predicted_shape = layer.compute_output_shape(tf_inputs.shape)
    self.assertEqual(outputs.shape, tuple(predicted_shape))

    # If not using channels_first, convert back to it to compare to SciPy.
    if data_format != "channels_first":
      outputs = np.moveaxis(outputs, -1, 1)

    # Compute the equivalent result using SciPy and compare.
    expected = self.scipy_convolve_valid(
        corr, inputs, kernel, strides_down, strides_up, extra_pad_end,
        channel_separable)
    self.assertAllClose(expected, outputs, rtol=0, atol=1e-3)

  def run_same(self, batch, input_support, channels, filters, kernel_support,
               corr, strides_down, strides_up, padding, extra_pad_end,
               channel_separable, data_format, activation, use_bias):
    assert channels == filters == 1

    # Create input array.
    input_shape = (batch, 1) + input_support
    inputs = np.arange(np.prod(input_shape))
    inputs = inputs.reshape(input_shape).astype(np.float32)
    if data_format != "channels_first":
      tf_inputs = tf.constant(np.moveaxis(inputs, 1, -1))
    else:
      tf_inputs = tf.constant(inputs)

    # Create kernel array. This is an identity kernel, so the outputs should
    # be equal to the inputs except for up- and downsampling.
    tf_kernel = initializers.IdentityInitializer()(
        shape=kernel_support + (1, filters), dtype=tf.float32)

    # Run SignalConv* layer.
    layer_class = {
        3: signal_conv.SignalConv1D,
        4: signal_conv.SignalConv2D,
        5: signal_conv.SignalConv3D,
    }[inputs.ndim]
    layer = layer_class(
        1, kernel_support, corr=corr, strides_down=strides_down,
        strides_up=strides_up, padding=padding, extra_pad_end=extra_pad_end,
        channel_separable=channel_separable, data_format=data_format,
        activation=activation, use_bias=use_bias, kernel_parameter=tf_kernel)
    outputs = layer(tf_inputs).numpy()

    # Check that SignalConv* computes the correct output size.
    predicted_shape = layer.compute_output_shape(tf_inputs.shape)
    self.assertEqual(outputs.shape, tuple(predicted_shape))

    # If not using channels_first, convert back to it to compare to input.
    if data_format != "channels_first":
      outputs = np.moveaxis(outputs, -1, 1)

    # Upsample and then downsample inputs.
    expected = inputs
    if not all(s == 1 for s in strides_up):
      expected = self.numpy_upsample(expected, strides_up, extra_pad_end)
    slices = (slice(None), slice(None))
    slices += tuple(slice(None, None, s) for s in strides_down)
    expected = expected[slices]

    self.assertAllClose(expected, outputs, rtol=0, atol=1e-3)

  def is_implemented(self, batch, input_support, channels, filters,
                     kernel_support, corr, strides_down, strides_up, padding,
                     extra_pad_end, channel_separable, data_format, activation,
                     use_bias):
    """Determine if SignalConv* implements the given arguments."""

    # If convolution is requested, or kernels can be flipped, we can use the
    # transpose ops.
    can_use_transpose = (
        not corr or all(s % 2 == 1 for s in kernel_support))

    # If upsampling is requested, or convolution and kernels can't be flipped,
    # we must use the transpose ops.
    must_use_transpose = (
        any(s != 1 for s in strides_up) or
        (not corr and any(s % 2 != 1 for s in kernel_support)))

    # If we must use transpose ops but can't, we fail.
    if must_use_transpose and not can_use_transpose:
      return False

    # Channel-separable is only implemented for 1D and 2D.
    if channel_separable and len(input_support) > 2:
      return False

    # Channel-separable with upsampling is only implemented for homogeneous
    # strides.
    if channel_separable and any(s != strides_up[0] for s in strides_up):
      return False

    # If we have to use the depthwise backprop op, we can't use filters > 1.
    if channel_separable and must_use_transpose and filters != 1:
      return False

    return True

  @property
  def data_formats(self):
    # On CPU, many ops don't support the channels first data format. Hence, if
    # no GPU is available, we skip these tests.
    if tf.config.experimental.list_physical_devices("GPU"):
      return ("channels_first", "channels_last")
    else:
      return ("channels_last",)

  def run_or_fail(self, method,
                  batch, input_support, channels, filters, kernel_support,
                  corr, strides_down, strides_up, padding, extra_pad_end,
                  channel_separable, data_format, activation, use_bias):
    args = dict(locals())
    del args["self"]
    del args["method"]
    if self.is_implemented(**args):
      try:
        method(**args)
      except:
        msg = []
        for k in sorted(args):
          msg.append(f"{k}={args[k]}")
        print("Failed when it shouldn't have: " + ", ".join(msg))
        raise
    else:
      try:
        with self.assertRaisesRegexp(NotImplementedError, "SignalConv"):
          method(**args)
      except:
        msg = []
        for k in sorted(args):
          msg.append(f"{k}={args[k]}")
        print("Did not fail when it should have: " + ", ".join(msg))
        raise

  def test_1d_valid_spatial(self):
    """Test 1D valid convolutions with different supports/strides."""
    batch = 1
    padding = "valid"
    channels = 1
    filters = 1
    activation = None
    use_bias = False
    for channel_separable in [False, True]:
      for input_support in [(12,), (7,)]:
        for kernel_support in [(1,), (2,), (7,)]:
          for corr in [False, True]:
            for strides_down, strides_up, extra_pad_end in zip(
                [(1,), (1,), (1,), (1,), (1,), (2,), (5,), (2,)],
                [(1,), (2,), (2,), (3,), (3,), (1,), (1,), (3,)],
                [True, False, True, False, True, True, True, True]):
              for data_format in self.data_formats:
                self.run_or_fail(
                    self.run_valid,
                    batch, input_support, channels, filters,
                    kernel_support, corr, strides_down, strides_up,
                    padding, extra_pad_end, channel_separable,
                    data_format, activation, use_bias)

  def test_1d_valid_channels(self):
    """Test 1D valid convolutions with multiple channels/filters."""
    batch = 1
    padding = "valid"
    input_support = (9,)
    kernel_support = (3,)
    corr = True
    strides_down = (1,)
    extra_pad_end = True
    activation = None
    use_bias = False
    for channel_separable in [False, True]:
      for channels, filters in zip([1, 2], [2, 1]):
        for strides_up in [(1,), (2,)]:
          for data_format in self.data_formats:
            self.run_or_fail(
                self.run_valid,
                batch, input_support, channels, filters,
                kernel_support, corr, strides_down, strides_up,
                padding, extra_pad_end, channel_separable,
                data_format, activation, use_bias)

  def test_1d_same_zeros_spatial(self):
    """Test 1D same_zeros convolutions with different supports/strides."""
    batch = 1
    padding = "same_zeros"
    channels = 1
    filters = 1
    channel_separable = False
    activation = None
    use_bias = False
    for input_support in [(12,), (7,)]:
      for kernel_support in [(1,), (2,), (3,), (7,)]:
        for corr in [False, True]:
          for strides_down, strides_up, extra_pad_end in zip(
              [(1,), (1,), (1,), (2,), (5,), (2,)],
              [(1,), (2,), (3,), (1,), (1,), (3,)],
              [True, False, True, True, True, True]):
            for data_format in self.data_formats:
              self.run_or_fail(
                  self.run_same,
                  batch, input_support, channels, filters,
                  kernel_support, corr, strides_down, strides_up,
                  padding, extra_pad_end, channel_separable,
                  data_format, activation, use_bias)

  def test_1d_same_padding(self):
    """Test 1D same convolutions with different padding modes."""
    batch = 1
    channels = 1
    filters = 1
    input_support = (8,)
    kernel_support = (3,)
    corr = True
    strides_up = (1,)
    strides_down = (1,)
    extra_pad_end = True
    channel_separable = False
    activation = None
    use_bias = False
    for padding in ["same_reflect"]:
      for data_format in self.data_formats:
        self.run_or_fail(
            self.run_same,
            batch, input_support, channels, filters,
            kernel_support, corr, strides_down, strides_up,
            padding, extra_pad_end, channel_separable,
            data_format, activation, use_bias)

  def test_1d_bias_activation(self):
    """Test 1D convolutions with bias and activation."""
    batch = 1
    channels = 1
    filters = 1
    input_support = (6,)
    kernel_support = (3,)
    corr = True
    strides_up = (1,)
    strides_down = (1,)
    extra_pad_end = True
    channel_separable = False
    activation = tf.identity
    use_bias = True
    padding = "valid"
    for data_format in self.data_formats:
      self.run_or_fail(
          self.run_valid,
          batch, input_support, channels, filters,
          kernel_support, corr, strides_down, strides_up,
          padding, extra_pad_end, channel_separable,
          data_format, activation, use_bias)

  def test_2d_valid_spatial(self):
    """Test 2D valid convolutions with different supports/strides."""
    batch = 1
    padding = "valid"
    channels = 1
    filters = 1
    activation = None
    use_bias = False
    for channel_separable in [False, True]:
      for input_support in [(10, 7), (5, 8)]:
        for kernel_support in [(5, 2), (2, 3), (3, 3)]:
          for corr in [False, True]:
            for strides_down, strides_up, extra_pad_end in zip(
                [(1, 1), (2, 2), (1, 1), (1, 1), (3, 5)],
                [(1, 1), (1, 1), (2, 2), (4, 3), (1, 1)],
                [True, True, False, True, True]):
              for data_format in self.data_formats:
                self.run_or_fail(
                    self.run_valid,
                    batch, input_support, channels, filters,
                    kernel_support, corr, strides_down, strides_up,
                    padding, extra_pad_end, channel_separable,
                    data_format, activation, use_bias)

  def test_2d_valid_channels(self):
    """Test 2D valid convolutions with multiple channels/filters."""
    batch = 1
    padding = "valid"
    input_support = (8, 7)
    kernel_support = (3, 3)
    corr = False
    strides_down = (1, 1)
    extra_pad_end = False
    activation = None
    use_bias = False
    for channel_separable in [False, True]:
      for channels, filters in zip([1, 2], [2, 1]):
        for strides_up in [(1, 1), (2, 2)]:
          for data_format in self.data_formats:
            self.run_or_fail(
                self.run_valid,
                batch, input_support, channels, filters,
                kernel_support, corr, strides_down, strides_up,
                padding, extra_pad_end, channel_separable,
                data_format, activation, use_bias)

  def test_2d_same_zeros_spatial(self):
    """Test 2D same_zeros convolutions with different supports/strides."""
    batch = 1
    padding = "same_zeros"
    channels = 1
    filters = 1
    channel_separable = False
    activation = None
    use_bias = False
    for input_support in [(4, 7), (5, 6)]:
      for kernel_support in [(3, 2), (2, 6), (3, 3)]:
        for corr in [False, True]:
          for strides_down, strides_up, extra_pad_end in zip(
              [(1, 1), (1, 1), (1, 1), (3, 5), (2, 3)],
              [(1, 1), (2, 3), (5, 2), (1, 1), (3, 2)],
              [True, False, True, True, False]):
            for data_format in self.data_formats:
              self.run_or_fail(
                  self.run_same,
                  batch, input_support, channels, filters,
                  kernel_support, corr, strides_down, strides_up,
                  padding, extra_pad_end, channel_separable,
                  data_format, activation, use_bias)

  def test_2d_same_padding(self):
    """Test 2D same convolutions with different padding modes."""
    batch = 1
    channels = 1
    filters = 1
    input_support = (4, 5)
    kernel_support = (3, 2)
    corr = True
    strides_up = (1, 1)
    strides_down = (1, 1)
    extra_pad_end = True
    channel_separable = False
    activation = None
    use_bias = False
    for padding in ["same_reflect"]:
      for data_format in self.data_formats:
        self.run_or_fail(
            self.run_same,
            batch, input_support, channels, filters,
            kernel_support, corr, strides_down, strides_up,
            padding, extra_pad_end, channel_separable,
            data_format, activation, use_bias)

  def test_2d_bias_activation(self):
    """Test 2D convolutions with bias and activation."""
    batch = 1
    channels = 1
    filters = 1
    input_support = (4, 6)
    kernel_support = (2, 2)
    corr = True
    strides_up = (1, 1)
    strides_down = (1, 1)
    extra_pad_end = True
    channel_separable = False
    activation = tf.identity
    use_bias = True
    padding = "valid"
    for data_format in self.data_formats:
      self.run_or_fail(
          self.run_valid,
          batch, input_support, channels, filters,
          kernel_support, corr, strides_down, strides_up,
          padding, extra_pad_end, channel_separable,
          data_format, activation, use_bias)

  def test_3d_valid_spatial(self):
    """Test 3D valid convolutions with different supports/strides."""
    batch = 1
    padding = "valid"
    channels = 1
    filters = 1
    channel_separable = False
    activation = None
    use_bias = False
    for input_support in [(8, 7, 3), (5, 6, 4)]:
      for kernel_support in [(1, 2, 3), (2, 1, 2), (3, 3, 3)]:
        for corr in [False, True]:
          for strides_down, strides_up, extra_pad_end in zip(
              [(1, 1, 1), (1, 1, 1), (1, 1, 1), (3, 5, 4), (2, 1, 1)],
              [(1, 1, 1), (1, 3, 2), (2, 4, 1), (1, 1, 1), (1, 1, 2)],
              [True, False, True, True]):
            for data_format in self.data_formats:
              self.run_or_fail(
                  self.run_valid,
                  batch, input_support, channels, filters,
                  kernel_support, corr, strides_down, strides_up,
                  padding, extra_pad_end, channel_separable,
                  data_format, activation, use_bias)

  def test_3d_valid_channels(self):
    """Test 3D valid convolutions with multiple channels/filters."""
    batch = 1
    padding = "valid"
    input_support = (7, 5, 4)
    kernel_support = (2, 3, 2)
    corr = False
    strides_down = (1, 1, 1)
    extra_pad_end = False
    channel_separable = False
    activation = None
    use_bias = False
    for channels, filters in zip([1, 2], [2, 1]):
      for strides_up in [(1, 1, 1), (1, 2, 2)]:
        for data_format in self.data_formats:
          self.run_or_fail(
              self.run_valid,
              batch, input_support, channels, filters,
              kernel_support, corr, strides_down, strides_up,
              padding, extra_pad_end, channel_separable,
              data_format, activation, use_bias)

  def test_3d_same_zeros_spatial(self):
    """Test 3D same_zeros convolutions with different supports/strides."""
    batch = 1
    padding = "same_zeros"
    channels = 1
    filters = 1
    channel_separable = False
    activation = None
    use_bias = False
    for input_support in [(4, 5, 4), (5, 6, 3)]:
      for kernel_support in [(1, 2, 3), (2, 1, 2), (3, 3, 3)]:
        for corr in [False, True]:
          for strides_down, strides_up, extra_pad_end in zip(
              [(1, 1, 1), (1, 1, 1), (1, 1, 1), (3, 5, 4)],
              [(1, 1, 1), (4, 3, 2), (2, 1, 3), (1, 1, 1)],
              [True, False, True, True]):
            for data_format in self.data_formats:
              self.run_or_fail(
                  self.run_same,
                  batch, input_support, channels, filters,
                  kernel_support, corr, strides_down, strides_up,
                  padding, extra_pad_end, channel_separable,
                  data_format, activation, use_bias)

  def test_3d_same_padding(self):
    """Test 3D same convolutions with different padding modes."""
    batch = 1
    channels = 1
    filters = 1
    input_support = (6, 6, 5)
    kernel_support = (3, 2, 2)
    corr = True
    strides_up = (1, 1, 1)
    strides_down = (1, 1, 1)
    extra_pad_end = True
    channel_separable = False
    activation = None
    use_bias = False
    for padding in ["same_reflect"]:
      for data_format in self.data_formats:
        self.run_or_fail(
            self.run_same,
            batch, input_support, channels, filters,
            kernel_support, corr, strides_down, strides_up,
            padding, extra_pad_end, channel_separable,
            data_format, activation, use_bias)

  def test_3d_bias_activation(self):
    """Test 3D convolutions with bias and activation."""
    batch = 1
    channels = 1
    filters = 1
    input_support = (7, 2, 4)
    kernel_support = (1, 2, 3)
    corr = True
    strides_up = (1, 1, 1)
    strides_down = (1, 1, 1)
    extra_pad_end = True
    channel_separable = False
    activation = tf.identity
    use_bias = True
    padding = "valid"
    for data_format in self.data_formats:
      self.run_or_fail(
          self.run_valid,
          batch, input_support, channels, filters,
          kernel_support, corr, strides_down, strides_up,
          padding, extra_pad_end, channel_separable,
          data_format, activation, use_bias)


if __name__ == "__main__":
  tf.test.main()
