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

import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp

from tensorflow_compression.python.distributions import uniform_noise
from tensorflow_compression.python.entropy_models.continuous_batched import ContinuousBatchedEntropyModel


class ContinuousBatchedEntropyModelTest(tf.test.TestCase):

  def test_can_instantiate(self):
    noisy = uniform_noise.NoisyNormal(loc=0., scale=1.)
    em = ContinuousBatchedEntropyModel(noisy, 1)
    self.assertIs(em.distribution, noisy)
    self.assertEqual(em.coding_rank, 1)
    self.assertEqual(em.likelihood_bound, 1e-9)
    self.assertEqual(em.tail_mass, 2**-8)
    self.assertEqual(em.range_coder_precision, 12)
    self.assertEqual(em.dtype, noisy.dtype)
    self.assertEqual(em.quantization_offset(), 0)
    self.assertEqual(em.upper_tail(), 2.885635)
    self.assertEqual(em.lower_tail(), -2.885635)

  def test_requires_scalar_distributions(self):
    noisy = uniform_noise.UniformNoiseAdapter(
        tfp.distributions.MultivariateNormalDiag(
            loc=[-3, .2], scale_diag=[1, 2]))
    with self.assertRaises(ValueError):
      ContinuousBatchedEntropyModel(noisy, 1)

  def test_requires_coding_rank_bigger_than_distribution_batch_rank(self):
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

  def test_compression_consistent_with_quantization(self):
    noisy = uniform_noise.NoisyNormal(loc=.25, scale=10.)
    em = ContinuousBatchedEntropyModel(noisy, 1)
    x = noisy.base.sample([100])
    x_quantized = em.quantize(x)
    x_decompressed = em.decompress(em.compress(x), [100])
    self.assertAllEqual(x_decompressed, x_quantized)

  def test_information_bounds(self):
    # `bits(training=True)` should be greater than `bits(training=False)`
    # because it is defined as an upper bound (albeit for infinite data). The
    # actual length of the bit string should always be greater than
    # `bits(training=False)` because range coding is only asymptotically
    # optimal, and because it operates on quantized probabilities.
    for scale in 2 ** tf.linspace(-2., 7., 10):
      noisy = uniform_noise.NoisyNormal(loc=0., scale=scale)
      em = ContinuousBatchedEntropyModel(noisy, 1)
      x = noisy.base.sample([10000])
      bits_eval = em.bits(x, training=False)
      bits_training = em.bits(x, training=True)
      bits_compressed = 8 * len(em.compress(x).numpy())
      self.assertGreater(bits_training, .9975 * bits_eval)
      self.assertGreater(bits_compressed, bits_eval)

  def test_low_entropy_bounds(self):
    # For low entropy distributions, the training bound should be very loose,
    # and the overhead of range coding manageable.
    noisy = uniform_noise.NoisyNormal(loc=0., scale=.25)
    em = ContinuousBatchedEntropyModel(noisy, 1)
    x = noisy.base.sample([10000])
    bits_eval = em.bits(x, training=False)
    bits_training = em.bits(x, training=True)
    bits_compressed = 8 * len(em.compress(x).numpy())
    self.assertAllClose(bits_training, bits_eval, atol=0, rtol=1.25)
    self.assertAllClose(bits_compressed, bits_eval, atol=0, rtol=5e-3)

  def test_high_entropy_bounds(self):
    # For high entropy distributions, the training bound should be very tight,
    # and the overhead of range coding manageable.
    noisy = uniform_noise.NoisyNormal(loc=0., scale=100.)
    em = ContinuousBatchedEntropyModel(noisy, 1)
    x = noisy.base.sample([10000])
    bits_eval = em.bits(x, training=False)
    bits_training = em.bits(x, training=True)
    bits_compressed = 8 * len(em.compress(x).numpy())
    self.assertAllClose(bits_training, bits_eval, atol=0, rtol=5e-5)
    self.assertAllClose(bits_compressed, bits_eval, atol=0, rtol=5e-3)


if __name__ == "__main__":
  tf.test.main()
