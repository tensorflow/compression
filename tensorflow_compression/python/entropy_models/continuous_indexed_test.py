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

import tensorflow as tf
from tensorflow_compression.python.distributions import uniform_noise
from tensorflow_compression.python.entropy_models import continuous_indexed


# TODO(jonycgn): add further unit tests.


class ContinuousIndexedEntropyModelTest(tf.test.TestCase):

  def test_can_instantiate_one_dimensional(self):
    em = continuous_indexed.ContinuousIndexedEntropyModel(
        uniform_noise.NoisyNormal, (64,),
        dict(loc=lambda _: 0, scale=lambda i: tf.exp(i / 8 - 5)), 1,
        compression=True, channel_axis=None)
    self.assertIsInstance(em.prior, uniform_noise.NoisyNormal)
    self.assertEqual(em.coding_rank, 1)
    self.assertEqual(em.tail_mass, 2**-8)
    self.assertEqual(em.range_coder_precision, 12)
    self.assertEqual(em.dtype, tf.float32)
    x = tf.random.stateless_normal((3, 8, 16), seed=(0, 0))
    indexes = tf.cast(64 * tf.random.stateless_uniform((3, 8, 16), seed=(0, 0)),
                      tf.int32)
    em(x, indexes)
    bitstring = em.compress(x, indexes)
    x_hat = em.decompress(bitstring, indexes)
    self.assertAllLess(x - x_hat, 0.5)
    self.assertAllGreater(x - x_hat, -0.5)

  def test_can_instantiate_and_compress_n_dimensional(self):
    em = continuous_indexed.ContinuousIndexedEntropyModel(
        uniform_noise.NoisyLogisticMixture,
        (10, 10, 5),
        dict(
            loc=lambda i: i[..., 0:2] - 5,
            scale=lambda _: 1,
            weight=lambda i: tf.nn.softmax((i[..., 2:3] - 2) * [-1, 1]),
        ),
        1,
        compression=True
    )
    self.assertIsInstance(em.prior, uniform_noise.NoisyLogisticMixture)
    self.assertEqual(em.coding_rank, 1)
    self.assertEqual(em.channel_axis, -1)
    self.assertEqual(em.tail_mass, 2**-8)
    self.assertEqual(em.range_coder_precision, 12)
    self.assertEqual(em.dtype, tf.float32)
    x = tf.random.stateless_normal((3, 8, 16), seed=(0, 0))
    indexes = tf.cast(
        10 * tf.random.stateless_uniform((3, 8, 16, 3), seed=(0, 0)), tf.int32)
    em(x, indexes)
    bitstring = em.compress(x, indexes)
    x_hat = em.decompress(bitstring, indexes)
    self.assertAllLess(x - x_hat, 0.5)
    self.assertAllGreater(x - x_hat, -0.5)


class LocationScaleIndexedEntropyModelTest(tf.test.TestCase):

  def test_can_instantiate_and_compress(self):
    em = continuous_indexed.LocationScaleIndexedEntropyModel(
        uniform_noise.NoisyNormal,
        64,
        lambda i: tf.exp(i / 8 - 5),
        1,
        compression=True)
    self.assertIsInstance(em.prior, uniform_noise.NoisyNormal)
    self.assertEqual(em.coding_rank, 1)
    self.assertEqual(em.tail_mass, 2**-8)
    self.assertEqual(em.range_coder_precision, 12)
    self.assertEqual(em.dtype, tf.float32)
    x = tf.random.stateless_normal((3, 8, 16), seed=(0, 0))
    indexes = tf.cast(10 * tf.random.stateless_uniform((3, 8, 16), seed=(0, 0)),
                      tf.int32)
    loc = tf.random.stateless_uniform((3, 8, 16), seed=(0, 0))
    em(x, indexes, loc=loc)
    bitstring = em.compress(x, indexes, loc=loc)
    x_hat = em.decompress(bitstring, indexes, loc=loc)
    self.assertAllLessEqual(x - x_hat, 0.5)
    self.assertAllGreaterEqual(x - x_hat, -0.5)


if __name__ == "__main__":
  tf.test.main()
