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
"""Tests of uniform noise adapter distribution."""

import tensorflow as tf
from tensorflow_compression.python.distributions import helpers
from tensorflow_compression.python.distributions import uniform_noise


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
    # approach a unit-width uniform distribution. As a side effect, this tests
    # that `mean()` is defined (because not for all distributions the mean
    # coincides with the location parameter).
    dist = self.dist_cls(loc=10, scale=1e-7)
    mean = dist.mean()
    x = tf.linspace(mean - 1, mean + 1, 10)
    self.assertAllClose(dist.prob(x), [0, 0, 0, 1, 1, 1, 1, 0, 0, 0])

  def test_sampling_works(self):
    dist = self.dist_cls(loc=0, scale=[3, 5])
    sample = dist.sample((5, 4))
    self.assertEqual(sample.shape, (5, 4, 2))

  def test_tails_and_offset_are_in_order(self):
    dist = self.dist_cls(loc=10.3, scale=1.5)
    offset = helpers.quantization_offset(dist)
    lower_tail = helpers.lower_tail(dist, 2**-8)
    upper_tail = helpers.upper_tail(dist, 2**-8)
    self.assertGreater(upper_tail, lower_tail)
    self.assertAllClose(offset, 0.3)

  def test_stats_throw_error(self):
    dist = self.dist_cls(loc=1, scale=2)
    with self.assertRaises(NotImplementedError):
      dist.mode()
    with self.assertRaises(NotImplementedError):
      dist.quantile(.5)
    with self.assertRaises(NotImplementedError):
      dist.survival_function(.5)


class NoisyNormalTest(LocationScaleTest, tf.test.TestCase):

  dist_cls = uniform_noise.NoisyNormal


class NoisyLogisticTest(LocationScaleTest, tf.test.TestCase):

  dist_cls = uniform_noise.NoisyLogistic


class MixtureTest:
  """Common tests for noisy mixture distributions."""

  def test_can_instantiate_scalar(self):
    dist = self.dist_cls(loc=[3., -3.], scale=[5., 2.5], weight=[.3, .7])
    self.assertEqual(dist.batch_shape, ())
    self.assertEqual(dist.event_shape, ())

  def test_can_instantiate_batched(self):
    dist = self.dist_cls(
        loc=[[3., -3.], [2., -2.]], scale=[5., 2.5], weight=[.3, .7])
    self.assertEqual(dist.batch_shape, (2,))
    self.assertEqual(dist.event_shape, ())

  def test_variables_receive_gradients(self):
    loc = tf.Variable(tf.ones([2], dtype=tf.float32))
    log_scale = tf.Variable(tf.zeros([2], dtype=tf.float32))
    logit_weight = tf.Variable(tf.constant([.3, .7], dtype=tf.float32))
    with tf.GradientTape() as tape:
      dist = self.dist_cls(
          loc=loc, scale=tf.exp(log_scale), weight=tf.nn.softmax(logit_weight))
      x = tf.random.normal([20])
      loss = -tf.reduce_mean(dist.log_prob(x))
    grads = tape.gradient(loss, [loc, log_scale, logit_weight])
    self.assertLen(grads, 3)
    self.assertNotIn(None, grads)

  def test_uniform_is_special_case(self):
    # With the scale parameters going to zero, the adapted distribution should
    # approach a mixture of unit-width uniform distributions.
    dist = self.dist_cls(loc=[2.5, -1.], scale=[1e-7, 1e-7], weight=[.3, .7])
    mean = dist.components_distribution.mean()
    x = tf.linspace(mean[0] - 1, mean[0] + 1, 10)
    self.assertAllClose(dist.prob(x), [0, 0, 0, .3, .3, .3, .3, 0, 0, 0])
    x = tf.linspace(mean[1] - 1, mean[1] + 1, 10)
    self.assertAllClose(dist.prob(x), [0, 0, 0, .7, .7, .7, .7, 0, 0, 0])

  def test_sampling_works(self):
    dist = self.dist_cls(loc=[[0]], scale=[3, 5], weight=[.2, .8])
    sample = dist.sample((5, 4))
    self.assertEqual(sample.shape, (5, 4, 1))

  def test_tails_and_offset_are_in_order(self):
    dist = self.dist_cls(loc=[5.4, 8.6], scale=[1.4, 2], weight=[.6, .4])
    offset = helpers.quantization_offset(dist)
    lower_tail = helpers.lower_tail(dist, 2**-8)
    upper_tail = helpers.upper_tail(dist, 2**-8)
    self.assertGreater(upper_tail, lower_tail)
    self.assertAllClose(offset, 0.4)  # Decimal part of the peakiest mode (5.4).

  def test_stats_throw_error(self):
    dist = self.dist_cls(loc=[1, 0], scale=2, weight=[.1, .9])
    with self.assertRaises(NotImplementedError):
      dist.mode()
    with self.assertRaises(NotImplementedError):
      dist.quantile(.5)
    with self.assertRaises(NotImplementedError):
      dist.survival_function(.5)


class NoisyNormalMixtureTest(MixtureTest, tf.test.TestCase):

  dist_cls = uniform_noise.NoisyNormalMixture


class NoisyLogisticMixtureTest(MixtureTest, tf.test.TestCase):

  dist_cls = uniform_noise.NoisyLogisticMixture


if __name__ == "__main__":
  tf.test.main()
