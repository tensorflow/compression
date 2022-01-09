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
"""Tests for round adapters."""

from absl.testing import parameterized
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_compression.python.distributions import deep_factorized
from tensorflow_compression.python.distributions import round_adapters


def _test_log_prob_gradient_is_bounded(self, dist_cls, values, params=()):
  x = tf.constant(values)
  with tf.GradientTape(persistent=True) as g:
    df = dist_cls(**dict(params))
    g.watch(x)
    p = df.log_prob(x)
    # When this bound triggers, we expect the gradient to be zero.
    idx = (p < -32.0)
    p = tf.maximum(p, -32.0)
  dx = g.gradient(p, x)
  self.assertAllEqual(dx[idx], np.zeros_like(dx[idx]))
  self.assertTrue(np.all(np.isfinite(dx)), f"dx has non-finite value: {dx}")


class AdaptersTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      ("softround_deepfactorized",
       lambda d: round_adapters.SoftRoundAdapter(d, alpha=5.0),
       deep_factorized.DeepFactorized),
      ("softround_logistic",
       lambda d: round_adapters.SoftRoundAdapter(d, alpha=5.0),
       lambda: tfp.distributions.Logistic(loc=10.3, scale=1.5)),
      ("softround_normal",
       lambda d: round_adapters.SoftRoundAdapter(d, alpha=4.0),
       lambda: tfp.distributions.Normal(loc=10.4, scale=1.5)),
      ("noisysoftround_deepfactorized",
       lambda d: round_adapters.NoisySoftRoundAdapter(d, alpha=5.0),
       deep_factorized.DeepFactorized),
      ("noisysoftround_logistic",
       lambda d: round_adapters.NoisySoftRoundAdapter(d, alpha=5.0),
       lambda: tfp.distributions.Logistic(loc=10, scale=1.5)),
      ("noisysoftround_normal",
       lambda d: round_adapters.NoisySoftRoundAdapter(d, alpha=5.0),
       lambda: tfp.distributions.Normal(loc=10, scale=1.5)),
      ("round_deepfactorized",
       round_adapters.RoundAdapter,
       lambda: deep_factorized.DeepFactorized(init_scale=1.0)),
      ("round_logistic",
       round_adapters.RoundAdapter,
       lambda: tfp.distributions.Logistic(loc=1.5, scale=1.5)),
      ("round_normal",
       round_adapters.RoundAdapter,
       lambda: tfp.distributions.Normal(loc=1.5, scale=1.5)),
      ("noisyround_deepfactorized",
       round_adapters.NoisyRoundAdapter,
       lambda: deep_factorized.DeepFactorized(init_scale=1.0)),
      ("noisyround_logistic",
       round_adapters.NoisyRoundAdapter,
       lambda: tfp.distributions.Logistic(loc=1.5, scale=1.5)),
      ("noisyround_normal",
       round_adapters.NoisyRoundAdapter,
       lambda: tfp.distributions.Normal(loc=1.5, scale=1.5)),
      )
  def test_tails(self, adapter, distribution):
    dist = adapter(distribution())
    lower_tail = dist._lower_tail(2**-8)
    try:
      left_mass = dist.cdf(lower_tail)
    except NotImplementedError:
      # We use base distribution as a proxy for the tail mass.
      left_mass = dist.base.cdf(lower_tail)
    self.assertLessEqual(left_mass, 2**-8)

    upper_tail = dist._upper_tail(2**-8)
    try:
      right_mass = dist.survival_function(upper_tail)
    except NotImplementedError:
      # We use base distribution as a proxy for the tail mass.
      right_mass = dist.base.survival_function(upper_tail)
    self.assertLessEqual(right_mass, 2**-8)

    self.assertGreater(upper_tail, lower_tail)

  @parameterized.named_parameters(
      ("softround_logistic",
       lambda d: round_adapters.SoftRoundAdapter(d, alpha=5.0),
       lambda: tfp.distributions.Logistic(loc=10, scale=1.5)),
      ("softround_normal",
       lambda d: round_adapters.SoftRoundAdapter(d, alpha=5.0),
       lambda: tfp.distributions.Normal(loc=10, scale=1.5)),
      )
  def test_mode_and_quantile(self, adapter, distribution):
    dist = adapter(distribution())
    mode = dist.mode()
    left_mass = dist.cdf(mode)
    self.assertAllClose(left_mass, 0.5)
    quantile_75p = dist.quantile(0.75)
    left_mass = dist.cdf(quantile_75p)
    self.assertAllClose(left_mass, 0.75)

  def test_lacking_mode_and_quantile(self):
    dist = round_adapters.RoundAdapter(
        tfp.distributions.Logistic(loc=1.5, scale=1.5))
    with self.assertRaises(NotImplementedError):
      dist.mode()
    with self.assertRaises(NotImplementedError):
      dist.quantile(0.75)

  def test_lacking_tails_and_offset(self):
    class NonInvertibleAdapater(round_adapters.MonotonicAdapter):

      invertible = False

      def transform(self, x):
        return tf.ceil(x)

      def inverse_transform(self, y):
        return tf.floor(y)

    dist = NonInvertibleAdapater(tfp.distributions.Normal(loc=1.5, scale=1.5))
    with self.assertRaises(NotImplementedError):
      dist._lower_tail(0.01)
    with self.assertRaises(NotImplementedError):
      dist._upper_tail(0.01)


class NoisySoftRoundedDeepFactorizedTest(tf.test.TestCase):

  def test_uniform_is_special_case(self):
    # With the scale parameter going to zero, the density should approach a
    # unit-width uniform distribution.
    df = round_adapters.NoisySoftRoundedDeepFactorized(init_scale=1e-3)
    x = tf.linspace(-1., 1., 10)
    self.assertAllClose(df.prob(x), [0, 0, 0, 1, 1, 1, 1, 0, 0, 0])

  def test_log_prob_gradient_is_bounded(self):
    _test_log_prob_gradient_is_bounded(
        self,
        round_adapters.NoisySoftRoundedDeepFactorized,
        values=[0.0, 1.0, 2.0, 1e3])

  def test_log_prob_gradient_is_bounded_failcase(self):
    with self.assertRaises(AssertionError):
      # TODO(relational): Here we obtain NaN gradients due to instabilities
      # in the implementation of `log_prob` for the distribution, and
      # we should ideally fix this.
      _test_log_prob_gradient_is_bounded(
          self,
          round_adapters.NoisySoftRoundedDeepFactorized,
          values=[1e6, 1e9])


class LocationScaleTest:
  """Common tests for noisy location-scale family of distributions."""

  def test_can_instantiate_scalar(self):
    dist = self.dist_cls(loc=3., scale=5.)
    self.assertEqual(dist.batch_shape, ())
    self.assertEqual(dist.event_shape, ())

  def test_can_instantiate_batched(self):
    dist = self.dist_cls(loc=[3., 2.], scale=5.)
    self.assertEqual(dist.batch_shape, (2,))
    self.assertEqual(dist.event_shape, ())

  def test_variables_receive_gradients(self):
    loc = tf.Variable(1., dtype=tf.float32)
    log_scale = tf.Variable(0., dtype=tf.float32)
    with tf.GradientTape() as tape:
      dist = self.dist_cls(loc=loc, scale=tf.exp(log_scale))
      x = tf.random.normal([20])
      loss = -tf.reduce_mean(dist.log_prob(x))
    grads = tape.gradient(loss, [loc, log_scale])
    self.assertLen(grads, 2)
    self.assertNotIn(None, grads)

  def test_uniform_is_special_case(self):
    # With the scale parameter going to zero, the adapted distribution should
    # approach a unit-width uniform distribution.
    dist = self.dist_cls(loc=5.0, scale=1e-7)
    x = tf.linspace(5.0 - 1, 5.0 + 1, 10)
    self.assertAllClose(dist.prob(x), [0, 0, 0, 1, 1, 1, 1, 0, 0, 0])

  def test_sampling_works(self):
    dist = self.dist_cls(loc=0, scale=[3, 5])
    sample = dist.sample((5, 4))
    self.assertEqual(sample.shape, (5, 4, 2))

  def test_tails_are_in_order(self):
    dist = self.dist_cls(loc=10, scale=1.5)
    lower_tail = dist._lower_tail(2**-8)
    upper_tail = dist._upper_tail(2**-8)
    self.assertGreater(upper_tail, lower_tail)

  def test_stats_throw_error(self):
    dist = self.dist_cls(loc=1, scale=2)
    with self.assertRaises(NotImplementedError):
      dist.mode()
    with self.assertRaises(NotImplementedError):
      dist.quantile(.5)
    with self.assertRaises(NotImplementedError):
      dist.survival_function(.5)


class NoisyRoundedNormalTest(tf.test.TestCase, LocationScaleTest):

  dist_cls = round_adapters.NoisyRoundedNormal


class NoisySoftRoundedNormalTest(tf.test.TestCase, LocationScaleTest):

  dist_cls = round_adapters.NoisySoftRoundedNormal

  def test_log_prob_gradient_is_bounded(self):
    _test_log_prob_gradient_is_bounded(
        self,
        round_adapters.NoisySoftRoundedNormal,
        values=[0.0, 1.0, 2.0, 1e3],
        params=dict(loc=0.0, scale=1.0))

  def test_log_prob_gradient_is_bounded_failcase(self):
    with self.assertRaises(AssertionError):
      # TODO(relational): Here we obtain NaN gradients due to instabilities
      # in the implementation of `log_prob` for the distribution, and
      # we should ideally fix this.
      _test_log_prob_gradient_is_bounded(
          self,
          round_adapters.NoisySoftRoundedNormal,
          values=[1e6, 1e9],
          params=dict(loc=0.0, scale=1.0))


if __name__ == "__main__":
  tf.test.main()
