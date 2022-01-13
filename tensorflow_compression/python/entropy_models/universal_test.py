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
"""Tests of universal entropy models."""

from absl.testing import parameterized
import scipy.stats
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_compression.python.distributions import deep_factorized
from tensorflow_compression.python.distributions import uniform_noise
from tensorflow_compression.python.entropy_models import universal


class UniversalBatchedEntropyModelTest(tf.test.TestCase,
                                       parameterized.TestCase):

  def test_can_instantiate_and_compress(self):
    prior = deep_factorized.NoisyDeepFactorized(batch_shape=(4, 4))
    em = universal.UniversalBatchedEntropyModel(
        prior, coding_rank=3, compression=True)
    x = tf.random.stateless_normal((3, 8, 4, 4), seed=(0, 0))
    bitstring = em.compress(x)
    em(x)
    x_hat = em.decompress(bitstring, (8,))
    # Quantization noise should be between -.5 and .5
    u = x - x_hat
    self.assertAllLessEqual(tf.abs(u), 0.5)

  @parameterized.named_parameters(
      ("", True, "deep_factorized"),
      ("eval", False, "deep_factorized"),
      ("normal", True, "normal"),
      ("eval_normal", False, "normal"),
  )
  def test_bitstring_length_matches_estimates(self, training, prior):
    priors = {
        "deep_factorized":
            deep_factorized.NoisyDeepFactorized(batch_shape=(16,)),
        "normal":
            uniform_noise.NoisyNormal(loc=tf.range(16.0), scale=1.0)
    }
    prior = priors[prior]
    em = universal.UniversalBatchedEntropyModel(
        prior, coding_rank=2, compression=True)
    num_symbols = 1000
    # Source distribution is fixed as gaussian.
    source = priors["normal"].base
    x = source.sample((3, num_symbols), seed=0)
    x_perturbed, bits_estimate = em(x, training=training)
    bitstring = em.compress(x)
    x_decoded = em.decompress(bitstring, (num_symbols,))
    bitstring_bits = tf.reshape(
        [len(b) * 8 for b in bitstring.numpy().flatten()], bitstring.shape)
    # Max error 1% and 2 bytes.
    self.assertAllClose(bits_estimate, bitstring_bits, atol=16, rtol=0.01)
    # Quantization noise should be between -.5 and .5
    self.assertAllLessEqual(tf.abs(x - x_decoded), 0.5)
    self.assertAllLessEqual(tf.abs(x - x_perturbed), 0.5)

  def test_bitstring_length_matches_entropy_normal(self, scale=1e-8):
    prior = uniform_noise.NoisyNormal(loc=100 * tf.range(15.0), scale=scale)
    base_df = prior.base
    em = universal.UniversalBatchedEntropyModel(
        prior, coding_rank=2, compression=True)
    num_samples = 100000
    x = base_df.sample(num_samples, seed=0)
    bitstring = em.compress(x)
    x_decoded = em.decompress(bitstring, (num_samples,))
    bits = len(bitstring.numpy()) * 8
    bits_per_sample = bits / num_samples
    # Quantization noise should be between -.5 and .5
    self.assertAllLessEqual(tf.abs(x - x_decoded), 0.5)

    # Lets estimate entropy via sampling the distribution.
    samples = prior.sample(num_samples, seed=0)
    log_probs = prior.log_prob(samples) / tf.math.log(2.0)
    entropy_bits = -tf.reduce_sum(log_probs)
    rtol = 0.01  # Maximum relative error 1%.
    atol = 16  # Maximum 2 bytes absolute error.
    self.assertLessEqual(bits_per_sample, entropy_bits * (1 + rtol) + atol)

  def test_laplace_tail_mass_for_large_inputs(self):
    prior = deep_factorized.NoisyDeepFactorized(batch_shape=(1,))
    em = universal.UniversalBatchedEntropyModel(
        prior,
        coding_rank=1,
        compression=True,
        laplace_tail_mass=1e-3)
    x = tf.convert_to_tensor([1e3, 1e4, 1e5, 1e6, 1e7, 1e8], tf.float32)
    _, bits = em(x[..., None])
    self.assertAllClose(bits, tf.abs(x) / tf.math.log(2.0), rtol=0.01)

  def test_laplace_tail_mass_for_small_inputs(self):
    prior = deep_factorized.NoisyDeepFactorized(batch_shape=(1,))
    em1 = universal.UniversalBatchedEntropyModel(
        prior,
        coding_rank=1,
        compression=True,
        laplace_tail_mass=1e-3)
    em2 = universal.UniversalBatchedEntropyModel(
        prior,
        coding_rank=1,
        compression=True)
    x = tf.linspace(-10.0, 10.0, 50)
    _, bits1 = em1(x[..., None])
    _, bits2 = em2(x[..., None])
    self.assertAllClose(bits1, bits2, rtol=0.01, atol=0.05)

  def test_expected_grads_gives_gradients(self):
    priors = {
        "deep_factorized":
            deep_factorized.NoisyDeepFactorized(batch_shape=(16,)),
        "normal":
            uniform_noise.NoisyNormal(loc=tf.range(16.0), scale=1.0)
    }
    prior = priors["deep_factorized"]
    em = universal.UniversalBatchedEntropyModel(
        prior, coding_rank=2, compression=True, expected_grads=True)
    self.assertTrue(em._expected_grads)
    num_symbols = 1000
    # Source distribution is fixed as gaussian.
    source = priors["normal"].base
    x = source.sample((3, num_symbols), seed=0)
    with tf.GradientTape(persistent=True) as g:
      g.watch(x)
      x2, bits = em(x, training=True)
    self.assertIsInstance(g.gradient(x2, x), tf.Tensor)
    self.assertIsInstance(g.gradient(bits, x), tf.Tensor)
    for variable in em.trainable_variables:
      self.assertIsInstance(g.gradient(bits, variable), tf.Tensor)


class UniversalIndexedEntropyModelTest(tf.test.TestCase,
                                       parameterized.TestCase):

  def setUp(self):
    super().setUp()
    tf.random.set_seed(1234)

  def test_can_instantiate_n_dimensional(self):
    em = universal.UniversalIndexedEntropyModel(
        uniform_noise.NoisyLogisticMixture,
        index_ranges=(10, 10, 5),
        parameter_fns=dict(
            loc=lambda i: i[..., 0:2] - 5,
            scale=lambda _: 1,
            weight=lambda i: tf.nn.softmax((i[..., 2:3] - 2) * [-1, 1]),
        ),
        coding_rank=1,
    )
    self.assertEqual(em.coding_rank, 1)
    self.assertEqual(em._laplace_tail_mass, 0.0)
    self.assertEqual(em.tail_mass, 2**-8)
    self.assertEqual(em.dtype, tf.float32)

  def test_can_instantiate_and_compress_n_dimensional(self):
    em = universal.UniversalIndexedEntropyModel(
        uniform_noise.NoisyLogisticMixture,
        index_ranges=(10, 10, 5),
        parameter_fns=dict(
            loc=lambda i: i[..., 0:2] - 5,
            scale=lambda _: 1,
            weight=lambda i: tf.nn.softmax((i[..., 2:3] - 2) * [-1, 1]),
        ),
        coding_rank=1,
        compression=True)
    x = tf.random.stateless_normal((3, 8, 16), seed=(0, 0))
    indexes = tf.cast(
        10 * tf.random.stateless_uniform((3, 8, 16, 3), seed=(0, 0)), tf.int32)
    em(x, indexes)
    bitstring = em.compress(x, indexes)
    x_hat = em.decompress(bitstring, indexes)
    # Quantization noise should be between -.5 and .5
    u = x - x_hat
    self.assertAllLessEqual(tf.abs(u), 0.5)

  def test_accurate_predictions_give_small_bitstring_length(self):
    # If we can perfectly predict locations with a very small scale, the
    # bitstring_length should be very small.
    em = universal.UniversalIndexedEntropyModel(
        prior_fn=uniform_noise.NoisyNormal,
        index_ranges=(10, 10),
        parameter_fns=dict(
            loc=lambda i: i[..., 0] / 3.0,
            scale=lambda i: tf.exp(-i[..., 1] - 20),  # Very small scale.
        ),
        coding_rank=1,
        compression=True)
    num_symbols = 1000
    # Random indices in the valid index ranges.
    indexes = tf.cast(
        tf.random.stateless_uniform((3, num_symbols, 16, 2),
                                    minval=0.0,
                                    maxval=10.0,
                                    seed=(0, 0)), tf.int32)
    prior = em._make_prior(indexes)
    x = prior.base.sample((), seed=0)
    y = prior.base.sample((), seed=0)
    self.assertAllClose(x, y)
    bitstring = em.compress(x, indexes)
    bitstring_bits = tf.reshape(
        [len(b) * 8 for b in bitstring.numpy().flatten()], bitstring.shape)
    # Maximum 2 bytes.
    self.assertAllLessEqual(bitstring_bits, 16.0)
    x_decoded = em.decompress(bitstring, indexes)
    # Maximum error is .5
    self.assertAllLessEqual(tf.abs(x - x_decoded), 0.5)

  @parameterized.named_parameters(
      ("", True),
      ("eval", False),
  )
  def test_bitstring_length_matches_estimates(self, training):
    em = universal.UniversalIndexedEntropyModel(
        prior_fn=uniform_noise.NoisyNormal,
        index_ranges=(10, 10),
        parameter_fns=dict(
            loc=lambda i: (i[..., 0] - 5.0) / 10.0,
            scale=lambda i: tf.exp(-i[..., 1]),
        ),
        coding_rank=1,
        compression=True)
    self.assertEqual(em.coding_rank, 1)
    self.assertEqual(em._laplace_tail_mass, 0.0)
    self.assertEqual(em.tail_mass, 2**-8)
    self.assertEqual(em.range_coder_precision, 12)
    self.assertEqual(em.dtype, tf.float32)
    num_symbols = 1000
    # Source distribution is gaussian with stddev 1.
    x = tf.random.stateless_normal((1, 1, num_symbols), seed=(0, 0))
    # We predict the distribution correctly (mean = 5-5=0, scale = exp(0)=1.
    indexes = tf.concat((5 * tf.ones((1, 1, num_symbols, 1)), 0 * tf.ones(
        (1, 1, num_symbols, 1))),
                        axis=-1)
    x_perturbed, bits_estimate = em(x, indexes, training=training)
    bitstring = em.compress(x, indexes)
    x_decoded = em.decompress(bitstring, indexes)
    bitstring_bits = tf.reshape(
        [len(b) * 8 for b in bitstring.numpy().flatten()], bitstring.shape)
    # Max error 1% and 1 byte.
    self.assertAllClose(bits_estimate, bitstring_bits, atol=8, rtol=0.01)
    # Quantization noise should be between -.5 and .5
    self.assertAllLessEqual(tf.abs(x - x_decoded), 0.5)
    self.assertAllLessEqual(tf.abs(x - x_perturbed), 0.5)

  def _test_bitstring_length_matches_entropy(self, prior_fn, base_prior_fn,
                                             scale):
    em = universal.UniversalIndexedEntropyModel(
        prior_fn=prior_fn,
        index_ranges=(10, 10),
        parameter_fns=dict(
            loc=lambda i: (i[..., 0] - 5.0) / 10.0,
            scale=lambda i: scale * tf.exp(-i[..., 1]),
        ),
        coding_rank=1,
        compression=True,
        tail_mass=1e-5,
        num_noise_levels=15)
    num_symbols = 10000
    base_df = base_prior_fn(loc=0.0, scale=scale)
    x = base_df.sample((1, 1, num_symbols), seed=0)
    # We predict the distribution correctly.
    indexes = tf.concat((5 * tf.ones((1, 1, num_symbols, 1)), 0 * tf.ones(
        (1, 1, num_symbols, 1))),
                        axis=-1)
    bitstring = em.compress(x, indexes)
    bits = len(bitstring.numpy()[0, 0]) * 8
    bits_per_symbol = bits / num_symbols

    df = prior_fn(loc=0.0, scale=scale)

    # Lets estimate entropy via sampling the distribution.
    samples = df.sample(num_symbols, seed=0)
    log_probs = df.log_prob(samples) / tf.math.log(2.0)
    entropy = -tf.reduce_mean(log_probs)
    rtol = 0.01  # Maximum relative error 1%.
    atol = 16 / num_symbols  # Maximum 2 bytes absolute error.
    self.assertLessEqual(bits_per_symbol, entropy * (1 + rtol) + atol)

  @parameterized.named_parameters(
      *((str(s), float(s)) for s in ["1e-8", "1e-4", "1e-2", "1e-1", 1, 3, 5]))
  def test_bitstring_length_matches_entropy_normal(self, scale):
    self._test_bitstring_length_matches_entropy(uniform_noise.NoisyNormal,
                                                tfp.distributions.Normal, scale)

  @parameterized.named_parameters(
      *((str(s), float(s)) for s in ["1e-8", "1e-4", "1e-2", "1e-1", 1, 3, 5]))
  def test_bitstring_length_matches_entropy_logistic(self, scale):
    self._test_bitstring_length_matches_entropy(uniform_noise.NoisyLogistic,
                                                tfp.distributions.Logistic,
                                                scale)

  def _test_bits_estimate_matches_entropy(self, prior_fn, base_prior_fn, scale):
    em = universal.UniversalIndexedEntropyModel(
        prior_fn=prior_fn,
        index_ranges=(10, 10),
        parameter_fns=dict(
            loc=lambda i: (i[..., 0] - 5.0) / 10.0,
            scale=lambda i: scale * tf.exp(-i[..., 1]),
        ),
        coding_rank=1,
        compression=True,
        tail_mass=1e-5,
        num_noise_levels=15)
    num_symbols = 1000000
    base_df = base_prior_fn(loc=0.1, scale=scale)
    x = base_df.sample((1, 1, num_symbols), seed=0)
    # We predict the distribution correctly.
    indexes = tf.concat((6 * tf.ones((1, 1, num_symbols, 1)), 0 * tf.ones(
        (1, 1, num_symbols, 1))),
                        axis=-1)
    bits = em(x, indexes)[1].numpy()[0, 0]
    bits_per_symbol = bits / num_symbols

    df = prior_fn(loc=0.0, scale=scale)

    # Lets estimate entropy via sampling the distribution.
    samples = df.sample(num_symbols, seed=0)
    log_probs = df.log_prob(samples) / tf.math.log(2.0)
    entropy = -tf.reduce_mean(log_probs)
    # Should be very close.
    self.assertAllClose(bits_per_symbol, entropy, rtol=0.001, atol=1e-3)

  @parameterized.named_parameters(
      *((str(s), float(s)) for s in ["1e-8", "1e-4", "1e-2", "1e-1", 1, 3, 5]))
  def test_bits_estimate_matches_entropy_normal(self, scale):
    self._test_bits_estimate_matches_entropy(uniform_noise.NoisyNormal,
                                             tfp.distributions.Normal, scale)

  @parameterized.named_parameters(
      *((str(s), float(s)) for s in ["1e-8", "1e-4", "1e-2", "1e-1", 1, 3, 5]))
  def test_bits_estimate_matches_entropy_logistic(self, scale):
    self._test_bits_estimate_matches_entropy(uniform_noise.NoisyLogistic,
                                             tfp.distributions.Logistic, scale)

  @parameterized.named_parameters(*((str(s), float(s)) for s in [1, 3, 5]))
  def test_quantization_noise_is_uniform(self, scale):
    em = universal.UniversalIndexedEntropyModel(
        prior_fn=uniform_noise.NoisyNormal,
        index_ranges=(10, 10),
        parameter_fns=dict(
            loc=lambda i: (i[..., 0] - 5.0) / 10.0,
            scale=lambda i: scale * tf.exp(-i[..., 1]),
        ),
        coding_rank=1,
        compression=True)
    num_symbols = 10000
    # Source distribution is gaussian with stddev `scale`.
    x = tf.random.stateless_normal((1, 1, num_symbols),
                                   stddev=scale,
                                   seed=(0, 0))
    # We predict the distribution correctly.
    indexes = tf.concat((5 * tf.ones((1, 1, num_symbols, 1)), 0 * tf.ones(
        (1, 1, num_symbols, 1))),
                        axis=-1)
    bitstring = em.compress(x, indexes)
    x_hat = em.decompress(bitstring, indexes)
    # Quantization noise should be between -.5 and .5
    u = x - x_hat
    self.assertAllLessEqual(tf.abs(u), 0.5)
    # Check distribution has right statistics.
    _, p = scipy.stats.kstest(u, "uniform", (-0.5, 1.0))
    self.assertGreater(p, 1e-6)

  def test_expected_grads_or_not_gives_same_bits(self):
    x = tf.random.stateless_normal((3, 10000, 16), seed=(0, 0))
    indexes = tf.cast(
        10 * tf.random.stateless_uniform((3, 10000, 16, 3), seed=(0, 0)),
        tf.int32)
    em_expected = universal.UniversalIndexedEntropyModel(
        uniform_noise.NoisyLogisticMixture,
        index_ranges=(10, 10, 5),
        parameter_fns=dict(
            loc=lambda i: i[..., 0:2] - 5,
            scale=lambda _: 1,
            weight=lambda i: tf.nn.softmax((i[..., 2:3] - 2) * [-1, 1]),
        ),
        coding_rank=2,
        expected_grads=True)
    self.assertTrue(em_expected._expected_grads)
    x_hat, bits_expected = em_expected(x, indexes)
    # Quantization noise should be between -.5 and .5
    u = x - x_hat
    self.assertAllLessEqual(tf.abs(u), 0.5)

    em_not_expected = universal.UniversalIndexedEntropyModel(
        uniform_noise.NoisyLogisticMixture,
        index_ranges=(10, 10, 5),
        parameter_fns=dict(
            loc=lambda i: i[..., 0:2] - 5,
            scale=lambda _: 1,
            weight=lambda i: tf.nn.softmax((i[..., 2:3] - 2) * [-1, 1]),
        ),
        coding_rank=2,
        expected_grads=False)
    self.assertFalse(em_not_expected._expected_grads)
    x_hat, bits_not_expected = em_not_expected(x, indexes)
    # Quantization noise should be between -.5 and .5
    u = x - x_hat
    self.assertAllLessEqual(tf.abs(u), 0.5)

    self.assertAllClose(bits_not_expected, bits_expected, rtol=0.001)

  def test_expected_grads_gives_gradients(self):
    x = tf.random.stateless_normal((3, 10000, 16), seed=(0, 0))
    indexes = tf.cast(10 * tf.random.stateless_uniform(
        (3, 10000, 16, 3), seed=(0, 0)), tf.float32)
    em = universal.UniversalIndexedEntropyModel(
        uniform_noise.NoisyLogisticMixture,
        index_ranges=(10, 10, 5),
        parameter_fns=dict(
            loc=lambda i: i[..., 0:2] - 5,
            scale=lambda _: 1,
            weight=lambda i: tf.nn.softmax((i[..., 2:3] - 2) * [-1, 1]),
        ),
        coding_rank=2,
        expected_grads=True)
    self.assertTrue(em._expected_grads)
    with tf.GradientTape(persistent=True) as g:
      g.watch((x, indexes))
      x2, bits = em(x, indexes)
    self.assertIsInstance(g.gradient(x2, x), tf.Tensor)
    self.assertIsInstance(g.gradient(bits, x), tf.Tensor)
    self.assertIsInstance(g.gradient(bits, indexes), tf.Tensor)
    self.assertIsNone(g.gradient(x2, indexes))

if __name__ == "__main__":
  tf.test.main()
