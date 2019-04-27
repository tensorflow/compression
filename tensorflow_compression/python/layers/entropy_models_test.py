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
"""Tests of entropy models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import scipy.stats
import tensorflow as tf

from tensorflow_compression.python.layers import entropy_models


class EntropyBottleneckTest(tf.test.TestCase):

  def test_noise(self):
    # Tests that the noise added is uniform noise between -0.5 and 0.5.
    inputs = tf.placeholder(tf.float32, (None, 1))
    layer = entropy_models.EntropyBottleneck()
    noisy, _ = layer(inputs, training=True)
    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      values = np.linspace(-50, 50, 100)[:, None]
      noisy, = sess.run([noisy], {inputs: values})
    self.assertFalse(np.allclose(values, noisy, rtol=0, atol=.45))
    self.assertAllClose(values, noisy, rtol=0, atol=.5)

  def test_quantization_init(self):
    # Tests that inputs are quantized to full integer values right after
    # initialization.
    inputs = tf.placeholder(tf.float32, (None, 1))
    layer = entropy_models.EntropyBottleneck()
    quantized, _ = layer(inputs, training=False)
    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      values = np.linspace(-50, 50, 100)[:, None]
      quantized, = sess.run([quantized], {inputs: values})
    self.assertAllClose(np.around(values), quantized, rtol=0, atol=1e-6)

  def test_quantization(self):
    # Tests that inputs are not quantized to full integer values after quantiles
    # have been updated. However, the difference between input and output should
    # be between -0.5 and 0.5, and the offset must be consistent.
    inputs = tf.placeholder(tf.float32, (None, 1))
    layer = entropy_models.EntropyBottleneck()
    quantized, _ = layer(inputs, training=False)
    opt = tf.train.GradientDescentOptimizer(learning_rate=1)
    self.assertEqual(1, len(layer.losses))
    step = opt.minimize(layer.losses[0])
    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      sess.run(step)
      values = np.linspace(-50, 50, 100)[:, None]
      quantized, = sess.run([quantized], {inputs: values})
    self.assertAllClose(values, quantized, rtol=0, atol=.5)
    diff = np.ravel(np.around(values) - quantized) % 1
    self.assertAllClose(diff, np.full_like(diff, diff[0]), rtol=0, atol=5e-6)
    self.assertNotEqual(diff[0], 0)

  def test_codec_init(self):
    # Tests that inputs are compressed and decompressed correctly, and quantized
    # to full integer values right after initialization.
    inputs = tf.placeholder(tf.float32, (1, None, 1))
    layer = entropy_models.EntropyBottleneck(
        data_format="channels_last", init_scale=30)
    bitstrings = layer.compress(inputs)
    decoded = layer.decompress(bitstrings, tf.shape(inputs)[1:])
    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      values = np.linspace(-50, 50, 100)[None, :, None]
      decoded, = sess.run([decoded], {inputs: values})
    self.assertAllClose(np.around(values), decoded, rtol=0, atol=1e-6)

  def test_codec(self):
    # Tests that inputs are compressed and decompressed correctly, and not
    # quantized to full integer values after quantiles have been updated.
    # However, the difference between input and output should be between -0.5
    # and 0.5, and the offset must be consistent.
    inputs = tf.placeholder(tf.float32, (1, None, 1))
    layer = entropy_models.EntropyBottleneck(
        data_format="channels_last", init_scale=40)
    bitstrings = layer.compress(inputs)
    decoded = layer.decompress(bitstrings, tf.shape(inputs)[1:])
    opt = tf.train.GradientDescentOptimizer(learning_rate=1)
    self.assertEqual(1, len(layer.losses))
    step = opt.minimize(layer.losses[0])
    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      sess.run(step)
      self.assertEqual(1, len(layer.updates))
      sess.run(layer.updates[0])
      values = np.linspace(-50, 50, 100)[None, :, None]
      decoded, = sess.run([decoded], {inputs: values})
    self.assertAllClose(values, decoded, rtol=0, atol=.5)
    diff = np.ravel(np.around(values) - decoded) % 1
    self.assertAllClose(diff, np.full_like(diff, diff[0]), rtol=0, atol=5e-6)
    self.assertNotEqual(diff[0], 0)

  def test_channels_last(self):
    # Test the layer with more than one channel and multiple input dimensions,
    # with the channels in the last dimension.
    inputs = tf.placeholder(tf.float32, (None, None, None, 2))
    layer = entropy_models.EntropyBottleneck(
        data_format="channels_last", init_scale=20)
    noisy, _ = layer(inputs, training=True)
    quantized, _ = layer(inputs, training=False)
    bitstrings = layer.compress(inputs)
    decoded = layer.decompress(bitstrings, tf.shape(inputs)[1:])
    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      self.assertEqual(1, len(layer.updates))
      sess.run(layer.updates[0])
      values = 5 * np.random.normal(size=(7, 5, 3, 2))
      noisy, quantized, decoded = sess.run(
          [noisy, quantized, decoded], {inputs: values})
    self.assertAllClose(values, noisy, rtol=0, atol=.5)
    self.assertAllClose(values, quantized, rtol=0, atol=.5)
    self.assertAllClose(values, decoded, rtol=0, atol=.5)

  def test_channels_first(self):
    # Test the layer with more than one channel and multiple input dimensions,
    # with the channel dimension right after the batch dimension.
    inputs = tf.placeholder(tf.float32, (None, 3, None, None))
    layer = entropy_models.EntropyBottleneck(
        data_format="channels_first", init_scale=10)
    noisy, _ = layer(inputs, training=True)
    quantized, _ = layer(inputs, training=False)
    bitstrings = layer.compress(inputs)
    decoded = layer.decompress(bitstrings, tf.shape(inputs)[1:])
    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      self.assertEqual(1, len(layer.updates))
      sess.run(layer.updates[0])
      values = 2.5 * np.random.normal(size=(2, 3, 5, 7))
      noisy, quantized, decoded = sess.run(
          [noisy, quantized, decoded], {inputs: values})
    self.assertAllClose(values, noisy, rtol=0, atol=.5)
    self.assertAllClose(values, quantized, rtol=0, atol=.5)
    self.assertAllClose(values, decoded, rtol=0, atol=.5)

  def test_compress(self):
    # Test compression and decompression, and produce test data for
    # `test_decompress`. If you set the constant at the end to `True`, this test
    # will fail and the log will contain the new test data.
    inputs = tf.placeholder(tf.float32, (2, 3, 9))
    layer = entropy_models.EntropyBottleneck(
        data_format="channels_first", filters=(), init_scale=2)
    bitstrings = layer.compress(inputs)
    decoded = layer.decompress(bitstrings, tf.shape(inputs)[1:])
    with self.test_session() as sess:
      values = 8 * np.random.uniform(size=(2, 3, 9)) - 4
      sess.run(tf.global_variables_initializer())
      self.assertEqual(1, len(layer.updates))
      sess.run(layer.updates[0])
      bitstrings, quantized_cdf, cdf_length, decoded = sess.run(
          [bitstrings, layer._quantized_cdf, layer._cdf_length, decoded],
          {inputs: values})
    self.assertAllClose(values, decoded, rtol=0, atol=.5)
    # Set this constant to `True` to log new test data for `test_decompress`.
    if False:  # pylint:disable=using-constant-test
      assert False, (bitstrings, quantized_cdf, cdf_length, decoded)

  # Data generated by `test_compress`.
  # pylint:disable=bad-whitespace,bad-continuation
  bitstrings = np.array([
      b"\x91\xf4\xdan2\xd3q\x97\xd0\x91N1~\xc4\xb0;\xd38\xa8\x90",
      b"?\xc7\xf9\x17\xa8\xcfu\x99\x1e4\xfe\xe0\xd3U`z\x15v",
  ], dtype=object)

  quantized_cdf = np.array([
      [    0,  5170, 11858, 19679, 27812, 35302, 65536],
      [    0,  6100, 13546, 21671, 29523, 36269, 65536],
      [    0,  6444, 14120, 22270, 29929, 36346, 65536],
  ], dtype=np.int32)

  cdf_length = np.array([7, 7, 7], dtype=np.int32)

  expected = np.array([
      [[-3.,  2.,  1., -3., -1., -3., -4., -2.,  2.],
       [-2.,  2.,  4.,  1.,  0., -3., -3.,  2.,  4.],
       [ 1.,  2.,  4., -1., -3.,  4.,  0., -2., -3.]],
      [[ 0.,  4.,  0.,  2.,  4.,  1., -2.,  1.,  4.],
       [ 2.,  2.,  3., -3.,  4., -1., -1.,  0., -1.],
       [ 3.,  0.,  3., -3.,  3.,  3., -3., -4., -1.]],
  ], dtype=np.float32)
  # pylint:enable=bad-whitespace,bad-continuation

  def test_decompress(self):
    # Test that decompression of values compressed with a previous version
    # works, i.e. that the file format doesn't change across revisions.
    bitstrings = tf.placeholder(tf.string)
    input_shape = tf.placeholder(tf.int32)
    quantized_cdf = tf.placeholder(tf.int32)
    cdf_length = tf.placeholder(tf.int32)
    layer = entropy_models.EntropyBottleneck(
        data_format="channels_first", filters=(), init_scale=2,
        dtype=tf.float32)
    layer.build(self.expected.shape)
    layer._quantized_cdf = quantized_cdf
    layer._cdf_length = cdf_length
    decoded = layer.decompress(bitstrings, input_shape[1:])
    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      decoded, = sess.run([decoded], {
          bitstrings: self.bitstrings, input_shape: self.expected.shape,
          quantized_cdf: self.quantized_cdf, cdf_length: self.cdf_length})
    self.assertAllClose(self.expected, decoded, rtol=0, atol=1e-6)

  def test_build_decompress(self):
    # Test that layer can be built when `decompress` is the first call to it.
    bitstrings = tf.placeholder(tf.string)
    input_shape = tf.placeholder(tf.int32, shape=[3])
    layer = entropy_models.EntropyBottleneck(dtype=tf.float32)
    layer.decompress(bitstrings, input_shape[1:], channels=5)
    self.assertTrue(layer.built)

  def test_normalization(self):
    # Test that densities are normalized correctly.
    inputs = tf.placeholder(tf.float32, (None, 1))
    layer = entropy_models.EntropyBottleneck(filters=(2,))
    _, likelihood = layer(inputs, training=True)
    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      x = np.repeat(np.arange(-200, 201), 2000)[:, None]
      likelihood, = sess.run([likelihood], {inputs: x})
    self.assertEqual(x.shape, likelihood.shape)
    integral = np.sum(likelihood) * .0005
    self.assertAllClose(1, integral, rtol=0, atol=1e-4)

  def test_entropy_estimates(self):
    # Test that entropy estimates match actual range coding.
    inputs = tf.placeholder(tf.float32, (1, None, 1))
    layer = entropy_models.EntropyBottleneck(
        filters=(2, 3), data_format="channels_last")
    _, likelihood = layer(inputs, training=True)
    diff_entropy = tf.reduce_sum(tf.log(likelihood)) / -np.log(2)
    _, likelihood = layer(inputs, training=False)
    disc_entropy = tf.reduce_sum(tf.log(likelihood)) / -np.log(2)
    bitstrings = layer.compress(inputs)
    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      self.assertEqual(1, len(layer.updates))
      sess.run(layer.updates[0])
      diff_entropy, disc_entropy, bitstrings = sess.run(
          [diff_entropy, disc_entropy, bitstrings],
          {inputs: np.random.normal(size=(1, 10000, 1))})
    codelength = 8 * sum(len(s) for s in bitstrings)
    self.assertAllClose(diff_entropy, disc_entropy, rtol=5e-3, atol=0)
    self.assertAllClose(disc_entropy, codelength, rtol=5e-3, atol=0)
    self.assertGreater(codelength, disc_entropy)


class SymmetricConditionalTest(object):

  def test_noise(self):
    # Tests that the noise added is uniform noise between -0.5 and 0.5.
    inputs = tf.placeholder(tf.float32, [None])
    scale = tf.placeholder(tf.float32, [None])
    layer = self.subclass(scale, [1])
    noisy, _ = layer(inputs, training=True)
    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      values = np.linspace(-50, 50, 100)
      noisy, = sess.run([noisy], {
          inputs: values,
          scale: np.random.uniform(1, 10, size=values.shape),
      })
    self.assertFalse(np.allclose(values, noisy, rtol=0, atol=.45))
    self.assertAllClose(values, noisy, rtol=0, atol=.5)

  def test_quantization(self):
    # Tests that inputs are quantized to full integer values.
    inputs = tf.placeholder(tf.float32, [None])
    scale = tf.placeholder(tf.float32, [None])
    layer = self.subclass(scale, [1], mean=None)
    quantized, _ = layer(inputs, training=False)
    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      values = np.linspace(-50, 50, 100)
      quantized, = sess.run([quantized], {
          inputs: values,
          scale: np.random.uniform(1, 10, size=values.shape),
      })
    self.assertAllClose(np.around(values), quantized, rtol=0, atol=1e-6)

  def test_quantization_mean(self):
    # Tests that inputs are quantized to integer values with a consistent offset
    # to the mean.
    inputs = tf.placeholder(tf.float32, [None])
    scale = tf.placeholder(tf.float32, [None])
    mean = tf.placeholder(tf.float32, [None])
    layer = self.subclass(scale, [1], mean=mean)
    quantized, _ = layer(inputs, training=False)
    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      values = np.linspace(-50, 50, 100)
      mean_values = np.random.normal(size=values.shape)
      quantized, = sess.run([quantized], {
          inputs: values,
          scale: np.random.uniform(1, 10, size=values.shape),
          mean: mean_values,
      })
    self.assertAllClose(
        np.around(values - mean_values) + mean_values, quantized,
        rtol=0, atol=1e-5)

  def test_codec(self):
    # Tests that inputs are compressed and decompressed correctly, and quantized
    # to full integer values.
    inputs = tf.placeholder(tf.float32, [None, None])
    scale = tf.placeholder(tf.float32, [None, None])
    layer = self.subclass(
        scale, [2 ** x for x in range(-10, 10)], mean=None)
    bitstrings = layer.compress(inputs)
    decoded = layer.decompress(bitstrings)
    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      values = np.linspace(-50, 50, 100)[None]
      decoded, = sess.run([decoded], {
          inputs: values,
          scale: np.random.uniform(25, 75, size=values.shape),
      })
    self.assertAllClose(np.around(values), decoded, rtol=0, atol=1e-6)

  def test_codec_mean(self):
    # Tests that inputs are compressed and decompressed correctly, and quantized
    # to integer values with an offset consistent with the mean.
    inputs = tf.placeholder(tf.float32, [None, None])
    scale = tf.placeholder(tf.float32, [None, None])
    mean = tf.placeholder(tf.float32, [None, None])
    layer = self.subclass(
        scale, [2 ** x for x in range(-10, 10)], mean=mean)
    bitstrings = layer.compress(inputs)
    decoded = layer.decompress(bitstrings)
    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      values = np.linspace(-50, 50, 100)[None]
      mean_values = np.random.normal(size=values.shape)
      decoded, = sess.run([decoded], {
          inputs: values,
          scale: np.random.uniform(25, 75, size=values.shape),
          mean: mean_values,
      })
    self.assertAllClose(
        np.around(values - mean_values) + mean_values, decoded,
        rtol=0, atol=1e-5)

  def test_multiple_dimensions(self):
    # Test the layer with more than one channel and multiple input dimensions.
    inputs = tf.placeholder(tf.float32, [None, None, None, None])
    scale = tf.placeholder(tf.float32, [None, None, None, None])
    layer = self.subclass(
        scale, [2 ** x for x in range(-10, 10)])
    noisy, _ = layer(inputs, training=True)
    quantized, _ = layer(inputs, training=False)
    bitstrings = layer.compress(inputs)
    decoded = layer.decompress(bitstrings)
    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      values = 10 * np.random.normal(size=(2, 5, 3, 7))
      noisy, quantized, decoded = sess.run(
          [noisy, quantized, decoded],
          {inputs: values, scale: np.random.uniform(5, 15, size=values.shape)})
    self.assertAllClose(values, noisy, rtol=0, atol=.5)
    self.assertAllClose(values, quantized, rtol=0, atol=.5)
    self.assertAllClose(values, decoded, rtol=0, atol=.5)

  def test_compress(self):
    # Test compression and decompression, and produce test data for
    # `test_decompress`. If you set the constant at the end to `True`, this test
    # will fail and the log will contain the new test data.
    shape = (2, 7)
    scale_table = [2 ** x for x in range(-5, 1)]
    inputs = tf.placeholder(tf.float32, shape)
    scale = tf.placeholder(tf.float32, shape)
    indexes = tf.placeholder(tf.int32, shape)
    layer = self.subclass(scale, scale_table, indexes=indexes)
    bitstrings = layer.compress(inputs)
    decoded = layer.decompress(bitstrings)
    with self.test_session() as sess:
      values = 8 * np.random.uniform(size=shape) - 4
      indexes = np.random.randint(
          0, len(scale_table), size=shape, dtype=np.int32)
      sess.run(tf.global_variables_initializer())
      bitstrings, quantized_cdf, cdf_length, decoded = sess.run(
          [bitstrings, layer._quantized_cdf, layer._cdf_length, decoded],
          {inputs: values, layer.indexes: indexes})
    self.assertAllClose(values, decoded, rtol=0, atol=.5)
    # Set this constant to `True` to log new test data for `test_decompress`.
    if False:  # pylint:disable=using-constant-test
      assert False, (bitstrings, indexes, quantized_cdf, cdf_length, decoded)

  def test_decompress(self):
    # Test that decompression of values compressed with a previous version
    # works, i.e. that the file format doesn't change across revisions.
    shape = (2, 7)
    scale_table = [2 ** x for x in range(-5, 1)]
    bitstrings = tf.placeholder(tf.string)
    scale = tf.placeholder(tf.float32, shape)
    indexes = tf.placeholder(tf.int32, shape)
    layer = self.subclass(
        scale, scale_table, indexes=indexes, dtype=tf.float32)
    decoded = layer.decompress(bitstrings)
    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      decoded, = sess.run([decoded], {
          bitstrings: self.bitstrings,
          layer.indexes: self.indexes,
          layer._quantized_cdf: self.quantized_cdf,
          layer._cdf_length: self.cdf_length})
    self.assertAllClose(self.expected, decoded, rtol=0, atol=1e-6)

  def test_build_decompress(self):
    # Test that layer can be built when `decompress` is the first call to it.
    bitstrings = tf.placeholder(tf.string)
    scale = tf.placeholder(tf.float32, [None, None, None])
    layer = self.subclass(
        scale, [2 ** x for x in range(-10, 10)], dtype=tf.float32)
    layer.decompress(bitstrings)
    self.assertTrue(layer.built)

  def test_quantile_function(self):
    # Test that quantile function inverts cumulative.
    scale = tf.placeholder(tf.float64, [None])
    layer = self.subclass(scale, [1], dtype=tf.float64)
    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      quantiles = np.array([1e-5, 1e-2, .1, .5, .6, .8])
      locations = layer._standardized_quantile(quantiles)
      locations = tf.constant(locations, tf.float64)
      values, = sess.run([layer._standardized_cumulative(locations)])
    self.assertAllClose(quantiles, values, rtol=1e-12, atol=0)

  def test_distribution(self):
    # Tests that the model represents the underlying distribution convolved
    # with a uniform.
    inputs = tf.placeholder(tf.float32, [None, None])
    scale = tf.placeholder(tf.float32, [None, None])
    layer = self.subclass(scale, [1], scale_bound=0, mean=None)
    _, likelihood = layer(inputs, training=False)
    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      values = np.arange(-5, 1)[:, None]  # must be integers due to quantization
      scales = 2 ** np.linspace(-3, 3, 10)[None, :]
      likelihoods, = sess.run([likelihood], {inputs: values, scale: scales})
    expected = (
        self.scipy_class.cdf(values + .5, scale=scales) -
        self.scipy_class.cdf(values - .5, scale=scales))
    self.assertAllClose(expected, likelihoods, rtol=1e-5, atol=1e-7)

  def test_entropy_estimates(self):
    # Test that analytical entropy, entropy estimates, and range coding match
    # each other.
    inputs = tf.placeholder(tf.float32, [None, None])
    scale = tf.placeholder(tf.float32, [None, None])
    layer = self.subclass(
        scale, [2 ** -10, 1, 10], scale_bound=0, likelihood_bound=0)
    _, likelihood = layer(inputs, training=True)
    diff_entropy = tf.reduce_mean(tf.log(likelihood), axis=1)
    diff_entropy /= -np.log(2)
    _, likelihood = layer(inputs, training=False)
    disc_entropy = tf.reduce_mean(tf.log(likelihood), axis=1)
    disc_entropy /= -np.log(2)
    bitstrings = layer.compress(inputs)
    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      scales = np.repeat([layer.scale_table], 10000, axis=0).T
      values = self.scipy_class.rvs(scale=scales, size=scales.shape)
      diff_entropy, disc_entropy, bitstrings = sess.run(
          [diff_entropy, disc_entropy, bitstrings],
          {inputs: values, scale: scales})
    codelength = [8 * len(s) for s in bitstrings]
    codelength = np.array(codelength) / values.shape[1]
    # The analytical entropy is only going to match the empirical for larger
    # scales because of the additive uniform noise. For scale values going to
    # zero, the empirical entropy will converge to zero (the entropy of a
    # standard uniform) instead of -infty. For large scale values, the additive
    # noise is negligible.
    theo_entropy = self.scipy_class.entropy(scale=10) / np.log(2)
    self.assertAllClose(0, diff_entropy[0], rtol=0, atol=5e-3)
    self.assertAllClose(theo_entropy, diff_entropy[-1], rtol=5e-3, atol=0)
    self.assertAllClose(diff_entropy, disc_entropy, rtol=5e-3, atol=5e-3)
    self.assertAllClose(disc_entropy, codelength, rtol=5e-3, atol=5e-3)
    # The range coder should have some overhead.
    self.assertTrue(all(codelength > disc_entropy))


class GaussianConditionalTest(tf.test.TestCase, SymmetricConditionalTest):

  subclass = entropy_models.GaussianConditional
  scipy_class = scipy.stats.norm

  # Data generated by `test_compress`.
  # pylint:disable=bad-whitespace,bad-continuation
  bitstrings = np.array([
      b"\xff\xff\x13\xff\xff\x0f\xff\xef\xa9\x000\xb9\xffT\x87\xffUB",
      b"\x10\xf1m-\xf0r\xac\x97\xb6\xd5",
  ], dtype=object)

  indexes = np.array([
      [1, 2, 3, 4, 2, 2, 1],
      [5, 5, 1, 5, 3, 2, 3],
  ], dtype=np.int32)

  quantized_cdf = np.array([
      [    0,     1, 65534, 65535, 65536,     0,     0,     0,     0],
      [    0,     1, 65534, 65535, 65536,     0,     0,     0,     0],
      [    0,     2, 65533, 65535, 65536,     0,     0,     0,     0],
      [    0,  1491, 64044, 65535, 65536,     0,     0,     0,     0],
      [    0,    88, 10397, 55138, 65447, 65535, 65536,     0,     0],
      [    0,   392,  4363, 20205, 45301, 61143, 65114, 65506, 65536],
  ], dtype=np.int32)

  cdf_length = np.array([5, 5, 5, 5, 7, 9], dtype=np.int32)

  expected = np.array([
      [-3.,  2.,  1., -3., -1., -3., -4.],
      [-2.,  2., -2.,  2.,  4.,  1.,  0.],
  ], dtype=np.float32)
  # pylint:enable=bad-whitespace,bad-continuation


class LogisticConditionalTest(tf.test.TestCase, SymmetricConditionalTest):

  subclass = entropy_models.LogisticConditional
  scipy_class = scipy.stats.logistic

  # Data generated by `test_compress`.
  # pylint:disable=bad-whitespace,bad-continuation
  bitstrings = np.array([
      b"\xff\xff\x13\xff\xff\x0e\x17\xfd\xb5B\x03\xff\xf4\x11",
      b",yh\x13)\x12F\xfb",
  ], dtype=object)

  indexes = np.array([
      [1, 2, 3, 4, 2, 2, 1],
      [5, 5, 1, 5, 3, 2, 3]
  ], dtype=np.int32)

  quantized_cdf = np.array([
      [    0,     1, 65534, 65535, 65536,     0,     0,     0,     0,
           0,     0,     0,     0,     0,     0,     0,     0],
      [    0,    22, 65513, 65535, 65536,     0,     0,     0,     0,
           0,     0,     0,     0,     0,     0,     0,     0],
      [    0,  1178, 64357, 65535, 65536,     0,     0,     0,     0,
           0,     0,     0,     0,     0,     0,     0,     0],
      [    0,   159,  7809, 57721, 65371, 65530, 65536,     0,     0,
           0,     0,     0,     0,     0,     0,     0,     0],
      [    0,    52,   431,  3100, 17617, 47903, 62420, 65089, 65468,
       65520, 65536,     0,     0,     0,     0,     0,     0],
      [    0,    62,   230,   683,  1884,  4935, 11919, 24706, 40758,
       53545, 60529, 63580, 64781, 65234, 65402, 65464, 65536],
  ], dtype=np.int32)

  cdf_length = np.array([ 5,  5,  5,  7, 11, 17], dtype=np.int32)

  expected = np.array([
      [-3.,  2.,  1., -3., -1., -3., -4.],
      [-2.,  2., -2.,  2.,  4.,  1.,  0.],
  ], dtype=np.float32)
  # pylint:enable=bad-whitespace,bad-continuation


class LaplacianConditionalTest(tf.test.TestCase, SymmetricConditionalTest):

  subclass = entropy_models.LaplacianConditional
  scipy_class = scipy.stats.laplace

  # Data generated by `test_compress`.
  # pylint:disable=bad-whitespace,bad-continuation
  bitstrings = np.array([
      b"\xff\xff\x13\xff\xff\x0e\xea\xc1\xd9n'\xff\xfe*",
      b"\x1b\x9c\xd3\x06\xde_\xc0$",
  ], dtype=object)

  indexes = np.array([
      [1, 2, 3, 4, 2, 2, 1],
      [5, 5, 1, 5, 3, 2, 3],
  ], dtype=np.int32)

  quantized_cdf = np.array([
      [    0,     1, 65534, 65535, 65536,     0,     0,     0,     0,
           0,     0,     0,     0,     0,     0],
      [    0,    11, 65524, 65535, 65536,     0,     0,     0,     0,
           0,     0,     0,     0,     0,     0],
      [    0,   600, 64935, 65535, 65536,     0,     0,     0,     0,
           0,     0,     0,     0,     0,     0],
      [    0,    80,  4433, 61100, 65453, 65533, 65536,     0,     0,
           0,     0,     0,     0,     0,     0],
      [    0,   191,  1602, 12025, 53451, 63874, 65285, 65476, 65536,
           0,     0,     0,     0,     0,     0],
      [    0,    85,   315,   940,  2640,  7262, 19825, 45612, 58175,
       62797, 64497, 65122, 65352, 65437, 65536],
  ], dtype=np.int32)

  cdf_length = np.array([ 5,  5,  5,  7,  9, 15], dtype=np.int32)

  expected = np.array([
      [-3.,  2.,  1., -3., -1., -3., -4.],
      [-2.,  2., -2.,  2.,  4.,  1.,  0.],
  ], dtype=np.float32)
  # pylint:enable=bad-whitespace,bad-continuation


if __name__ == "__main__":
  tf.test.main()
