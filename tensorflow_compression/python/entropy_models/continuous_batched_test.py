# Copyright 2020 Google LLC. All Rights Reserved.
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
"""Tests of batched continuous entropy model."""

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_compression.python.distributions import uniform_noise
from tensorflow_compression.python.entropy_models.continuous_batched import ContinuousBatchedEntropyModel


class ContinuousBatchedEntropyModelTest(tf.test.TestCase):

  def test_can_instantiate(self):
    noisy = uniform_noise.NoisyNormal(loc=0., scale=1.)
    em = ContinuousBatchedEntropyModel(noisy, 1)
    self.assertIs(em.prior, noisy)
    self.assertEqual(em.coding_rank, 1)
    self.assertEqual(em.tail_mass, 2**-8)
    self.assertEqual(em.range_coder_precision, 12)
    self.assertEqual(em.dtype, noisy.dtype)

  def test_can_instantiate_statelessly(self):
    noisy = uniform_noise.NoisyNormal(loc=.25, scale=1.)
    em = ContinuousBatchedEntropyModel(
        noisy, coding_rank=1, compression=True)
    self.assertEqual(em.compression, True)
    self.assertEqual(em.stateless, False)
    self.assertAllEqual(.25, em.quantization_offset)
    em = ContinuousBatchedEntropyModel(
        compression=True, stateless=True, coding_rank=1,
        prior_shape=noisy.batch_shape, dtype=noisy.dtype,
        cdf=em.cdf, cdf_offset=em.cdf_offset, cdf_length=em.cdf_length,
        quantization_offset=em.quantization_offset,
    )
    self.assertEqual(em.compression, True)
    self.assertEqual(em.stateless, True)
    self.assertAllEqual(.25, em.quantization_offset)
    with self.assertRaises(RuntimeError):
      em.prior  # pylint:disable=pointless-statement
    self.assertEqual(em.coding_rank, 1)
    self.assertEqual(em.tail_mass, 2**-8)
    self.assertEqual(em.range_coder_precision, 12)
    self.assertEqual(em.dtype, noisy.dtype)

  def test_requires_scalar_distributions(self):
    noisy = uniform_noise.UniformNoiseAdapter(
        tfp.distributions.MultivariateNormalDiag(
            loc=[-3, .2], scale_diag=[1, 2]))
    with self.assertRaises(ValueError):
      ContinuousBatchedEntropyModel(noisy, 1)

  def test_requires_coding_rank_bigger_than_prior_batch_rank(self):
    noisy = uniform_noise.NoisyLogistic(loc=0, scale=[[1], [2]])
    with self.assertRaises(ValueError):
      ContinuousBatchedEntropyModel(noisy, 0)
    with self.assertRaises(ValueError):
      ContinuousBatchedEntropyModel(noisy, 1)
    ContinuousBatchedEntropyModel(noisy, 2)
    ContinuousBatchedEntropyModel(noisy, 3)

  def test_quantizes_to_integers_modulo_offset(self):
    noisy = uniform_noise.NoisyNormal(loc=.25, scale=10.)
    em = ContinuousBatchedEntropyModel(noisy, 1)
    x = tf.range(-20., 20.) + .25
    x_perturbed = x + tf.random.uniform(x.shape, -.49, .49)
    x_quantized = em.quantize(x_perturbed)
    self.assertAllEqual(x, x_quantized)

  def test_gradients_are_straight_through(self):
    noisy = uniform_noise.NoisyNormal(loc=0, scale=1)
    em = ContinuousBatchedEntropyModel(noisy, 1)
    x = tf.range(-20., 20.)
    x_perturbed = x + tf.random.uniform(x.shape, -.49, .49)
    with tf.GradientTape() as tape:
      tape.watch(x_perturbed)
      x_quantized = em.quantize(x_perturbed)
    gradients = tape.gradient(x_quantized, x_perturbed)
    self.assertAllEqual(gradients, tf.ones_like(gradients))

  def test_default_kwargs_throw_error_on_compression(self):
    noisy = uniform_noise.NoisyNormal(loc=.25, scale=10.)
    em = ContinuousBatchedEntropyModel(noisy, 1)
    x = tf.zeros(10)
    with self.assertRaises(RuntimeError):
      em.compress(x)
    s = tf.zeros(10, dtype=tf.string)
    with self.assertRaises(RuntimeError):
      em.decompress(s, [10])

  def test_compression_consistent_with_quantization(self):
    noisy = uniform_noise.NoisyNormal(loc=.25, scale=10.)
    em = ContinuousBatchedEntropyModel(noisy, 1, compression=True)
    x = noisy.base.sample([100])
    x_quantized = em.quantize(x)
    x_decompressed = em.decompress(em.compress(x), [100])
    self.assertAllEqual(x_decompressed, x_quantized)

  def test_information_bounds(self):
    # bits w/ `training=True` should be greater than bits w/ `training=False`
    # because it is defined as an upper bound (albeit for infinite data). The
    # actual length of the bit string should always be greater than
    # bits w/ `training=False` because range coding is only asymptotically
    # optimal, and because it operates on quantized probabilities.
    for scale in 2 ** tf.linspace(-2., 7., 10):
      noisy = uniform_noise.NoisyNormal(loc=0., scale=scale)
      em = ContinuousBatchedEntropyModel(noisy, 1, compression=True)
      x = noisy.base.sample([10000])
      _, bits_eval = em(x, training=False)
      _, bits_training = em(x, training=True)
      bits_compressed = 8 * len(em.compress(x).numpy())
      self.assertGreater(bits_training, .9975 * bits_eval)
      self.assertGreater(bits_compressed, bits_eval)

  def test_low_entropy_bounds(self):
    # For low entropy distributions, the training bound should be very loose,
    # and the overhead of range coding manageable.
    noisy = uniform_noise.NoisyNormal(loc=0., scale=.25)
    em = ContinuousBatchedEntropyModel(noisy, 1, compression=True)
    x = noisy.base.sample([10000])
    _, bits_eval = em(x, training=False)
    _, bits_training = em(x, training=True)
    bits_compressed = 8 * len(em.compress(x).numpy())
    self.assertAllClose(bits_training, bits_eval, atol=0, rtol=1.25)
    self.assertAllClose(bits_compressed, bits_eval, atol=0, rtol=5e-3)

  def test_high_entropy_bounds(self):
    # For high entropy distributions, the training bound should be very tight,
    # and the overhead of range coding manageable.
    noisy = uniform_noise.NoisyNormal(loc=0., scale=100.)
    em = ContinuousBatchedEntropyModel(noisy, 1, compression=True)
    x = noisy.base.sample([10000])
    _, bits_eval = em(x, training=False)
    _, bits_training = em(x, training=True)
    bits_compressed = 8 * len(em.compress(x).numpy())
    self.assertAllClose(bits_training, bits_eval, atol=0, rtol=5e-5)
    self.assertAllClose(bits_compressed, bits_eval, atol=0, rtol=5e-3)

  def test_compression_works_after_serialization(self):
    noisy = uniform_noise.NoisyNormal(loc=.5, scale=8.)
    em = ContinuousBatchedEntropyModel(noisy, 1, compression=True)
    self.assertIsNot(em._quantization_offset, None)
    json = tf.keras.utils.serialize_keras_object(em)
    weights = em.get_weights()
    x = noisy.base.sample([100])
    x_quantized = em.quantize(x)
    x_compressed = em.compress(x)
    em = tf.keras.utils.deserialize_keras_object(json)
    em.set_weights(weights)
    self.assertAllEqual(em.compress(x), x_compressed)
    self.assertAllEqual(em.decompress(x_compressed, [100]), x_quantized)

  def test_compression_works_after_serialization_no_offset(self):
    noisy = uniform_noise.NoisyNormal(loc=0, scale=5.)
    em = ContinuousBatchedEntropyModel(noisy, 1, compression=True)
    self.assertIs(em._quantization_offset, None)
    json = tf.keras.utils.serialize_keras_object(em)
    weights = em.get_weights()
    x = noisy.base.sample([100])
    x_quantized = em.quantize(x)
    x_compressed = em.compress(x)
    em = tf.keras.utils.deserialize_keras_object(json)
    em.set_weights(weights)
    self.assertAllEqual(em.compress(x), x_compressed)
    self.assertAllEqual(em.decompress(x_compressed, [100]), x_quantized)

  def test_compression_works_in_tf_function(self):
    noisy = uniform_noise.NoisyNormal(loc=0, scale=5.)
    sample = noisy.base.sample([100])

    # Since tf.function traces each function twice, and only allows variable
    # creation in the first call, we need to have a stateful object in which we
    # create the entropy model only the first time the function is called, and
    # store it for the second time.

    class Compressor:

      def compress(self, values):
        if not hasattr(self, "em"):
          self.em = ContinuousBatchedEntropyModel(noisy, 1, compression=True)
        compressed = self.em.compress(values)
        decompressed = self.em.decompress(compressed, [])
        return decompressed

    values_eager = Compressor().compress(sample)
    values_function = tf.function(Compressor().compress)(sample)
    self.assertAllEqual(values_eager, values_function)

  def test_small_cdfs_for_dirac_prior_without_quantization_offset(self):
    prior = uniform_noise.NoisyNormal(loc=100 * tf.range(16.0), scale=1e-10)
    prior._quantization_offset = lambda: 0.0
    em = ContinuousBatchedEntropyModel(
        prior, coding_rank=2, compression=True)
    self.assertAllLessEqual(em._cdf_length, 10)

  def test_small_bitcost_for_dirac_prior(self):
    prior = uniform_noise.NoisyNormal(loc=100 * tf.range(16.0), scale=1e-10)
    em = ContinuousBatchedEntropyModel(
        prior, coding_rank=2, compression=True)
    num_symbols = 1000
    source = prior.base
    x = source.sample((3, num_symbols))
    _, bits_estimate = em(x, training=True)
    bitstring = em.compress(x)
    x_decoded = em.decompress(bitstring, (num_symbols,))
    bitstring_bits = tf.reshape(
        [len(b) * 8 for b in bitstring.numpy().flatten()], bitstring.shape)
    # Max 2 bytes.
    self.assertAllLessEqual(bits_estimate, 16)
    self.assertAllLessEqual(bitstring_bits, 16)
    # Quantization noise should be between -.5 and .5
    self.assertAllLessEqual(tf.abs(x - x_decoded), 0.5)


if __name__ == "__main__":
  tf.test.main()
