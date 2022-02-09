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
"""Tests of indexed continuous entropy model."""

from absl.testing import parameterized
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_compression.python.distributions import uniform_noise
from tensorflow_compression.python.entropy_models import continuous_indexed


class ContinuousIndexedEntropyModelTest(tf.test.TestCase,
                                        parameterized.TestCase):

  def get_model(self, prior_fn=uniform_noise.NoisyLogisticMixture,
                coding_rank=1, scale=1., **kwargs):
    return continuous_indexed.ContinuousIndexedEntropyModel(
        prior_fn,
        (2, 3, 5),
        dict(
            loc=lambda i: i[..., :2] - [0., 1.5],
            scale=lambda _: scale,
            weight=lambda i: tf.nn.softmax((i[..., 2:] - 2.) * [-1., 1.]),
        ),
        coding_rank,
        **kwargs)

  def get_samples(self, shape, scale=2., dtype=tf.float32):
    # This produces samples from a smoothed Laplacian with the requested scale.
    # They're not really samples from the prior, but approximately cover it,
    # and have the same tail behavior.
    x = tf.random.stateless_uniform(
        shape, minval=0., maxval=1., seed=(0, 1), dtype=dtype)
    s = tf.random.stateless_uniform(
        shape, minval=-1., maxval=1., seed=(1, 2), dtype=dtype)
    u = tf.random.stateless_uniform(
        shape, minval=-.5, maxval=.5, seed=(3, 4), dtype=dtype)
    x = (tf.math.log(x) * tf.math.sign(s) + u) * scale
    indexes = tf.random.stateless_uniform(
        tuple(shape) + (3,), minval=-.4, maxval=(2.4, 3.4, 5), seed=(5, 6),
        dtype=tf.float32)
    return x, indexes

  def test_can_instantiate_and_compress(self):
    em = self.get_model(compression=True)
    self.assertIsInstance(em.prior, uniform_noise.NoisyLogisticMixture)
    self.assertEqual(em.coding_rank, 1)
    self.assertEqual(em.channel_axis, -1)
    self.assertEqual(em.tail_mass, 2**-8)
    self.assertEqual(em.range_coder_precision, 12)
    self.assertEqual(em.bottleneck_dtype, tf.float32)
    self.assertEqual(em.prior.dtype, tf.float32)
    self.assertEqual(em.prior_dtype, tf.float32)
    x, indexes = self.get_samples((2, 5))
    x_tilde, bits = em(x, indexes)
    bitstring = em.compress(x, indexes)
    x_hat = em.decompress(bitstring, indexes)
    self.assertAllClose(x, x_hat, rtol=0, atol=.5)
    self.assertAllClose(x, x_tilde, rtol=0, atol=.5)
    self.assertEqual(bits.shape, (2,))
    self.assertAllGreaterEqual(bits, 0.)

  def test_can_instantiate_and_compress_statelessly(self):
    em = self.get_model(
        compression=True, stateless=True, prior_dtype=tf.float64)
    self.assertEqual(em.compression, True)
    self.assertEqual(em.stateless, True)
    self.assertIsInstance(em.prior, uniform_noise.NoisyLogisticMixture)
    self.assertEqual(em.coding_rank, 1)
    self.assertEqual(em.tail_mass, 2**-8)
    self.assertEqual(em.range_coder_precision, 12)
    self.assertEqual(em.bottleneck_dtype, tf.float32)
    self.assertEqual(em.prior.dtype, tf.float64)
    self.assertEqual(em.prior_dtype, tf.float64)
    x, indexes = self.get_samples((7,), dtype=tf.float32)
    x_tilde, bits = em(x, indexes)
    bitstring = em.compress(x, indexes)
    x_hat = em.decompress(bitstring, indexes)
    self.assertAllClose(x, x_hat, rtol=0, atol=.5)
    self.assertAllClose(x, x_tilde, rtol=0, atol=.5)
    self.assertEqual(bits.shape, ())
    self.assertAllGreaterEqual(bits, 0.)

  def test_indexes_are_clipped_correctly(self):
    em = self.get_model(compression=True, coding_rank=2)
    x, indexes = self.get_samples((7, 23))
    x_float_idx = em.decompress(em.compress(x, indexes), indexes)
    indexes = tf.cast(tf.round(indexes), tf.int32)
    x_int_idx = em.decompress(em.compress(x, indexes), indexes)
    self.assertAllEqual(x_float_idx, x_int_idx)
    self.assertAllClose(x, x_float_idx, rtol=0, atol=.5)

  def test_requires_scalar_distributions(self):
    def prior_fn(**_):
      return uniform_noise.UniformNoiseAdapter(
          tfp.distributions.MultivariateNormalDiag(
              loc=[-3, .2], scale_diag=[1, 2]))
    with self.assertRaises(ValueError):
      self.get_model(prior_fn=prior_fn, compression=True)

  def test_quantizes_to_integers(self):
    em = self.get_model()
    x = tf.range(-20., 20.)
    x_perturbed = x + tf.random.uniform(x.shape, -.49, .49)
    x_quantized = em.quantize(x_perturbed)
    self.assertAllEqual(x, x_quantized)

  def test_gradients_are_straight_through(self):
    em = self.get_model()
    x = tf.range(-20., 20.)
    x_perturbed = x + tf.random.uniform(x.shape, -.49, .49)
    with tf.GradientTape() as tape:
      tape.watch(x_perturbed)
      x_quantized = em.quantize(x_perturbed)
    gradients = tape.gradient(x_quantized, x_perturbed)
    self.assertAllEqual(gradients, tf.ones_like(gradients))

  def test_default_kwargs_throw_error_on_compression(self):
    em = self.get_model()
    x, indexes = self.get_samples((5,))
    with self.assertRaises(RuntimeError):
      em.compress(x, indexes)
    s = tf.zeros((), dtype=tf.string)
    with self.assertRaises(RuntimeError):
      em.decompress(s, indexes)

  def test_compression_consistent_with_quantization(self):
    em = self.get_model(compression=True)
    x, indexes = self.get_samples((100,))
    x_quantized = em.quantize(x)
    x_decompressed = em.decompress(em.compress(x, indexes), indexes)
    self.assertAllEqual(x_decompressed, x_quantized)

  @parameterized.parameters(*[2. ** i for i in range(-2, 8)])
  def test_information_bounds(self, scale):
    em = self.get_model(scale=scale, compression=True)
    x, indexes = self.get_samples([200000], scale=2. * scale)
    _, bits_eval = em(x, indexes, training=False)
    _, bits_training = em(x, indexes, training=True)
    bits_compressed = 8 * len(em.compress(x, indexes).numpy())
    # Asymptotically, the entropy estimate with `training=True` is an upper
    # bound on the entropy estimate with `training=False`. (With limited data,
    # fluctuations are possible.)
    with self.subTest("training bits > eval bits"):
      # Sample size is too small for the bound to be asymptotic. Increasing it
      # would make tests run too long.
      self.assertGreater(bits_training, 0.99999 * bits_eval)
    # Asymptotically, the length of the bit string should be greater than the
    # entropy estimate with `training=False` because range coding is only
    # asymptotically optimal, and because it operates on quantized
    # probabilities.
    with self.subTest("compressed bits > eval bits"):
      self.assertGreater(bits_compressed, bits_eval)
    # For low entropy distributions, the training bound can be very loose.
    if scale <= .5:
      with self.subTest("training bound loose"):
        self.assertAllClose(bits_training, bits_eval, atol=0, rtol=1e-2)
        self.assertNotAllClose(bits_training, bits_eval, atol=0, rtol=1e-4)
    # For high entropy distributions, the training bound should be tight.
    if scale >= 64:
      with self.subTest("training bound tight"):
        self.assertAllClose(bits_training, bits_eval, atol=0, rtol=1e-5)
    # The overhead of range coding should always be manageable.
    with self.subTest("range coding overhead"):
      self.assertAllClose(bits_compressed, bits_eval, atol=0, rtol=4e-2)

  def test_compression_works_in_tf_function(self):
    samples, indexes = self.get_samples((100,))

    # Since tf.function traces each function twice, and only allows variable
    # creation in the first call, we need to have a stateful object in which we
    # create the entropy model only the first time the function is called, and
    # store it for the second time.

    # We need this since `self` below shadows the test object.
    get_model = self.get_model

    class Compressor:

      def compress(self, values, indexes):
        if not hasattr(self, "em"):
          self.em = get_model(compression=True)
        compressed = self.em.compress(values, indexes)
        return self.em.decompress(compressed, indexes)

    values_eager = Compressor().compress(samples, indexes)
    values_function = tf.function(Compressor().compress)(samples, indexes)
    self.assertAllClose(samples, values_eager, rtol=0., atol=.5)
    self.assertAllEqual(values_eager, values_function)

  def test_dtypes_are_correct_with_mixed_precision(self):
    tf.keras.mixed_precision.set_global_policy("mixed_float16")
    try:
      em = self.get_model(compression=True, prior_dtype=tf.float64)
      self.assertIsInstance(em.prior, uniform_noise.NoisyLogisticMixture)
      self.assertEqual(em.bottleneck_dtype, tf.float16)
      self.assertEqual(em.prior.dtype, tf.float64)
      self.assertEqual(em.prior_dtype, tf.float64)
      x, indexes = self.get_samples((2, 5), dtype=tf.float16)
      x_tilde, bits = em(x, indexes)
      bitstring = em.compress(x, indexes)
      x_hat = em.decompress(bitstring, indexes)
      self.assertEqual(x_hat.dtype, tf.float16)
      self.assertAllClose(x, x_hat, rtol=0, atol=.5)
      self.assertEqual(x_tilde.dtype, tf.float16)
      self.assertAllClose(x, x_tilde, rtol=0, atol=.5)
      self.assertEqual(bits.dtype, tf.float64)
      self.assertEqual(bits.shape, (2,))
      self.assertAllGreaterEqual(bits, 0.)
    finally:
      tf.keras.mixed_precision.set_global_policy(None)


class LocationScaleIndexedEntropyModelTest(tf.test.TestCase):

  def get_model(self, prior_fn=uniform_noise.NoisyNormal,
                coding_rank=1, **kwargs):
    return continuous_indexed.LocationScaleIndexedEntropyModel(
        prior_fn,
        64,
        lambda i: tf.exp(i / 8. - 5.),
        coding_rank,
        **kwargs)

  def get_samples(self, shape):
    x = tf.random.stateless_normal(shape, stddev=5., seed=(0, 1))
    indexes = tf.random.stateless_uniform(
        shape, minval=-.4, maxval=64.4, seed=(0, 0), dtype=tf.float32)
    loc = tf.random.stateless_normal(shape, stddev=5., seed=(2, 3))
    return x, indexes, loc

  def test_can_instantiate_and_compress(self):
    em = self.get_model(compression=True)
    self.assertIsInstance(em.prior, uniform_noise.NoisyNormal)
    self.assertEqual(em.coding_rank, 1)
    self.assertEqual(em.tail_mass, 2**-8)
    self.assertEqual(em.range_coder_precision, 12)
    self.assertEqual(em.bottleneck_dtype, tf.float32)
    self.assertEqual(em.prior.dtype, tf.float32)
    self.assertEqual(em.prior_dtype, tf.float32)
    x, indexes, loc = self.get_samples((7, 4))
    x_tilde, bits = em(x, indexes, loc=loc)
    bitstring = em.compress(x, indexes, loc=loc)
    x_hat = em.decompress(bitstring, indexes, loc=loc)
    self.assertAllClose(x, x_hat, rtol=0, atol=.5)
    self.assertAllClose(x, x_tilde, rtol=0, atol=.5)
    self.assertEqual(bits.shape, (7,))
    self.assertAllGreaterEqual(bits, 0.)


if __name__ == "__main__":
  tf.test.main()
