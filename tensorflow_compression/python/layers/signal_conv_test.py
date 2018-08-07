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
"""Tests of signal processing convolution layers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports

import numpy as np
import scipy.signal
import tensorflow as tf

from tensorflow.python.platform import test

import tensorflow_compression as tfc


class SignalTest(tf.test.TestCase):

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
    tf_kernel = tfc.StaticParameterizer(tf.constant_initializer(kernel))

    # Run SignalConv* layer.
    layer_class = {
        3: tfc.SignalConv1D,
        4: tfc.SignalConv2D,
        5: tfc.SignalConv3D,
    }[inputs.ndim]
    layer = layer_class(
        filters, kernel_support, corr=corr, strides_down=strides_down,
        strides_up=strides_up, padding="valid", extra_pad_end=extra_pad_end,
        channel_separable=channel_separable, data_format=data_format,
        activation=activation, use_bias=use_bias,
        kernel_parameterizer=tf_kernel)
    tf_outputs = layer(tf_inputs)
    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      outputs = sess.run(tf_outputs)

    # Check that SignalConv* computes the correct output size.
    predicted_shape = layer.compute_output_shape(tf_inputs.shape)
    self.assertEqual(outputs.shape, tuple(predicted_shape.as_list()))

    # If not using channels_first, convert back to it to compare to SciPy.
    if data_format != "channels_first":
      outputs = np.moveaxis(outputs, -1, 1)

    # Compute the equivalent result using SciPy and compare.
    expected = self.scipy_convolve_valid(
        corr, inputs, kernel, strides_down, strides_up, extra_pad_end,
        channel_separable)
    self.assertAllClose(expected, outputs, rtol=0, atol=1e-6)

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
    tf_kernel = tfc.StaticParameterizer(tfc.IdentityInitializer())

    # Run SignalConv* layer.
    layer_class = {
        3: tfc.SignalConv1D,
        4: tfc.SignalConv2D,
        5: tfc.SignalConv3D,
    }[inputs.ndim]
    layer = layer_class(
        1, kernel_support, corr=corr, strides_down=strides_down,
        strides_up=strides_up, padding=padding, extra_pad_end=extra_pad_end,
        channel_separable=channel_separable, data_format=data_format,
        activation=activation, use_bias=use_bias,
        kernel_parameterizer=tf_kernel)
    tf_outputs = layer(tf_inputs)
    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      outputs = sess.run(tf_outputs)

    # Check that SignalConv* computes the correct output size.
    predicted_shape = layer.compute_output_shape(tf_inputs.shape)
    self.assertEqual(outputs.shape, tuple(predicted_shape.as_list()))

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

    self.assertAllClose(expected, outputs, rtol=0, atol=1e-6)

  def is_implemented(self, batch, input_support, channels, filters,
                     kernel_support, corr, strides_down, strides_up, padding,
                     extra_pad_end, channel_separable, data_format, activation,
                     use_bias):
    """Determine if SignalConv* implements the given arguments."""

    # Simultaneous up- and downsampling is not implemented.
    if (any(s != 1 for s in strides_down) and
        any(s != 1 for s in strides_up)):
      return False

    # Upsampled correlations are not implemented unless kernels are odd-length.
    if (corr and
        any(s != 1 for s in strides_up) and
        any(s % 2 != 1 for s in kernel_support)):
      return False

    # Downsampled convolutions are not implemented unless kernels are
    # odd-length.
    if (not corr and
        any(s != 1 for s in strides_down) and
        any(s % 2 != 1 for s in kernel_support)):
      return False

    # Channel-separable is only implemented for 2D and for homogeneous strides.
    if (channel_separable and
        (len(input_support) != 2 or
         any(s != strides_up[0] for s in strides_up) or
         any(s != strides_down[0] for s in strides_down))):
      return False

    # If any upsampling, or convolution with non-odd-length kernels is
    # requested, we have to use the depthwise backprop op. However, we can't
    # implement filters > 1 with it.
    if (channel_separable and
        (any(s != 1 for s in strides_up) or
         (not corr and any(s % 2 != 1 for s in kernel_support))) and
        filters != 1):
      return False

    return True

  @property
  def data_formats(self):
    # On CPU, many ops don't support the channels first data format. Hence, if
    # no GPU is available, we skip these tests.
    if test.is_gpu_available(cuda_only=True):
      return ("channels_first", "channels_last")
    else:
      return ("channels_last",)

  def test_1d_valid_spatial(self):
    """Test 1D valid convolutions with different supports/strides."""
    batch = 1
    padding = "valid"
    channels = 1
    filters = 1
    channel_separable = False
    activation = None
    use_bias = False
    for input_support in [(12,), (7,)]:
      for kernel_support in [(1,), (2,), (7,)]:
        for corr in [False, True]:
          for strides_down, strides_up, extra_pad_end in zip(
              [(1,), (1,), (1,), (1,), (1,), (2,), (5,), (2,)],
              [(1,), (2,), (2,), (3,), (3,), (1,), (1,), (3,)],
              [True, False, True, False, True, True, True, True]):
            for data_format in self.data_formats:
              args = (
                  batch, input_support, channels, filters,
                  kernel_support, corr, strides_down, strides_up,
                  padding, extra_pad_end, channel_separable,
                  data_format, activation, use_bias)
              if self.is_implemented(*args):
                try:
                  self.run_valid(*args)
                except:
                  print(*args)
                  raise
              else:
                with self.assertRaisesRegexp(
                    NotImplementedError, "SignalConv"):
                  self.run_valid(*args)

  def test_1d_valid_channels(self):
    """Test 1D valid convolutions with multiple channels/filters."""
    batch = 1
    padding = "valid"
    input_support = (9,)
    kernel_support = (3,)
    corr = True
    strides_down = (1,)
    extra_pad_end = True
    channel_separable = False
    activation = None
    use_bias = False
    for channels, filters in zip([1, 2], [2, 1]):
      for strides_up in [(1,), (2,)]:
        for data_format in self.data_formats:
          args = (
              batch, input_support, channels, filters,
              kernel_support, corr, strides_down, strides_up,
              padding, extra_pad_end, channel_separable,
              data_format, activation, use_bias)
          if self.is_implemented(*args):
            try:
              self.run_valid(*args)
            except:
              print(*args)
              raise
          else:
            with self.assertRaisesRegexp(
                NotImplementedError, "SignalConv"):
              self.run_valid(*args)

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
              args = (
                  batch, input_support, channels, filters,
                  kernel_support, corr, strides_down, strides_up,
                  padding, extra_pad_end, channel_separable,
                  data_format, activation, use_bias)
              if self.is_implemented(*args):
                try:
                  self.run_same(*args)
                except:
                  print(*args)
                  raise
              else:
                with self.assertRaisesRegexp(
                    NotImplementedError, "SignalConv"):
                  self.run_same(*args)

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
        args = (
            batch, input_support, channels, filters,
            kernel_support, corr, strides_down, strides_up,
            padding, extra_pad_end, channel_separable,
            data_format, activation, use_bias)
        if self.is_implemented(*args):
          try:
            self.run_same(*args)
          except:
            print(*args)
            raise
        else:
          with self.assertRaisesRegexp(
              NotImplementedError, "SignalConv"):
            self.run_same(*args)

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
      args = (
          batch, input_support, channels, filters,
          kernel_support, corr, strides_down, strides_up,
          padding, extra_pad_end, channel_separable,
          data_format, activation, use_bias)
      try:
        self.run_valid(*args)
      except:
        print(*args)
        raise

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
                args = (
                    batch, input_support, channels, filters,
                    kernel_support, corr, strides_down, strides_up,
                    padding, extra_pad_end, channel_separable,
                    data_format, activation, use_bias)
                if self.is_implemented(*args):
                  try:
                    self.run_valid(*args)
                  except:
                    print(*args)
                    raise
                else:
                  with self.assertRaisesRegexp(
                      NotImplementedError, "SignalConv"):
                    self.run_valid(*args)

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
            args = (
                batch, input_support, channels, filters,
                kernel_support, corr, strides_down, strides_up,
                padding, extra_pad_end, channel_separable,
                data_format, activation, use_bias)
            if self.is_implemented(*args):
              try:
                self.run_valid(*args)
              except:
                print(*args)
                raise
            else:
              with self.assertRaisesRegexp(
                  NotImplementedError, "SignalConv"):
                self.run_valid(*args)

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
              [(1, 1), (1, 1), (1, 1), (3, 5), (3, 5)],
              [(1, 1), (2, 3), (3, 2), (1, 1), (5, 3)],
              [True, False, True, True, True]):
            for data_format in self.data_formats:
              args = (
                  batch, input_support, channels, filters,
                  kernel_support, corr, strides_down, strides_up,
                  padding, extra_pad_end, channel_separable,
                  data_format, activation, use_bias)
              if self.is_implemented(*args):
                try:
                  self.run_same(*args)
                except:
                  print(*args)
                  raise
              else:
                with self.assertRaisesRegexp(
                    NotImplementedError, "SignalConv"):
                  self.run_same(*args)

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
    channel_separable = True
    activation = None
    use_bias = False
    for padding in ["same_reflect"]:
      for data_format in self.data_formats:
        args = (
            batch, input_support, channels, filters,
            kernel_support, corr, strides_down, strides_up,
            padding, extra_pad_end, channel_separable,
            data_format, activation, use_bias)
        if self.is_implemented(*args):
          try:
            self.run_same(*args)
          except:
            print(*args)
            raise
        else:
          with self.assertRaisesRegexp(
              NotImplementedError, "SignalConv"):
            self.run_same(*args)

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
    channel_separable = True
    activation = tf.identity
    use_bias = True
    padding = "valid"
    for data_format in self.data_formats:
      args = (
          batch, input_support, channels, filters,
          kernel_support, corr, strides_down, strides_up,
          padding, extra_pad_end, channel_separable,
          data_format, activation, use_bias)
      try:
        self.run_valid(*args)
      except:
        print(*args)
        raise

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
              args = (
                  batch, input_support, channels, filters,
                  kernel_support, corr, strides_down, strides_up,
                  padding, extra_pad_end, channel_separable,
                  data_format, activation, use_bias)
              if self.is_implemented(*args):
                try:
                  self.run_valid(*args)
                except:
                  print(*args)
                  raise
              else:
                with self.assertRaisesRegexp(
                    NotImplementedError, "SignalConv"):
                  self.run_valid(*args)

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
          args = (
              batch, input_support, channels, filters,
              kernel_support, corr, strides_down, strides_up,
              padding, extra_pad_end, channel_separable,
              data_format, activation, use_bias)
          if self.is_implemented(*args):
            try:
              self.run_valid(*args)
            except:
              print(*args)
              raise
          else:
            with self.assertRaisesRegexp(
                NotImplementedError, "SignalConv"):
              self.run_valid(*args)

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
              args = (
                  batch, input_support, channels, filters,
                  kernel_support, corr, strides_down, strides_up,
                  padding, extra_pad_end, channel_separable,
                  data_format, activation, use_bias)
              if self.is_implemented(*args):
                try:
                  self.run_same(*args)
                except:
                  print(*args)
                  raise
              else:
                with self.assertRaisesRegexp(
                    NotImplementedError, "SignalConv"):
                  self.run_same(*args)

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
        args = (
            batch, input_support, channels, filters,
            kernel_support, corr, strides_down, strides_up,
            padding, extra_pad_end, channel_separable,
            data_format, activation, use_bias)
        if self.is_implemented(*args):
          try:
            self.run_same(*args)
          except:
            print(*args)
            raise
        else:
          with self.assertRaisesRegexp(
              NotImplementedError, "SignalConv"):
            self.run_same(*args)

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
      args = (
          batch, input_support, channels, filters,
          kernel_support, corr, strides_down, strides_up,
          padding, extra_pad_end, channel_separable,
          data_format, activation, use_bias)
      try:
        self.run_valid(*args)
      except:
        print(*args)
        raise


if __name__ == "__main__":
  tf.test.main()
