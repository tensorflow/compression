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
"""Tests for soft round."""

from absl.testing import parameterized
import tensorflow as tf

from tensorflow_compression.python.ops import soft_round_ops


class SoftRoundTest(tf.test.TestCase, parameterized.TestCase):

  def test_soft_round_small_alpha_is_identity(self):
    x = tf.linspace(-2., 2., 50)
    y = soft_round_ops.soft_round(x, alpha=1e-13)
    self.assertAllClose(x, y)

  def test_soft_round_large_alpha_is_round(self):
    # We don't care what happens exactly near half-integer values:
    for offset in range(-5, 5):
      x = tf.linspace(offset - 0.499, offset + 0.499, 100)
      y = soft_round_ops.soft_round(x, alpha=2000.0)
      self.assertAllClose(tf.round(x), y, atol=0.02)

  def test_soft_inverse_round_small_alpha_is_identity(self):
    x = tf.linspace(-2., 2., 50)
    y = soft_round_ops.soft_round_inverse(x, alpha=1e-13)
    self.assertAllEqual(x, y)

  def test_soft_inverse_is_actual_inverse(self):
    x = tf.constant([-1.25, -0.75, 0.75, 1.25], dtype=tf.float32)
    y = soft_round_ops.soft_round(x, alpha=2.0)
    x2 = soft_round_ops.soft_round_inverse(y, alpha=2.0)
    self.assertAllClose(x, x2)

  def test_soft_round_inverse_large_alpha_is_ceil_minus_half(self):
    # We don't care what happens exactly near integer values:
    for offset in range(-5, 5):
      x = tf.linspace(offset + 0.001, offset + 0.999, 100)
      y = soft_round_ops.soft_round_inverse(x, alpha=5000.0)
      self.assertAllClose(tf.math.ceil(x) - 0.5, y, atol=0.001)

  def test_conditional_mean_large_alpha_is_round(self):
    # We don't care what happens exactly near integer values:
    for offset in range(-5, 5):
      x = tf.linspace(offset + 0.001, offset + 0.999, 100)
      y = soft_round_ops.soft_round_conditional_mean(x, alpha=5000.0)
      self.assertAllClose(tf.math.round(x), y, atol=0.001)

  @parameterized.parameters(0., 1e-6, 1e-2, 5., 1e6)
  def test_soft_round_values_and_gradients_are_finite(self, alpha):
    x = tf.linspace(0., 1., 11)  # covers exact integers and half-integers
    with tf.GradientTape() as tape:
      tape.watch(x)
      y = soft_round_ops.soft_round(x, alpha=alpha)
    dy = tape.gradient(y, x)
    self.assertAllEqual(tf.math.is_finite(y), tf.ones(x.shape, dtype=bool))
    self.assertAllEqual(tf.math.is_finite(dy), tf.ones(x.shape, dtype=bool))

  @parameterized.parameters(0., 1e-6, 1e-2, 5., 1e6)
  def test_soft_round_inverse_values_and_gradients_are_finite(self, alpha):
    x = tf.linspace(-.5, .5, 11)  # covers exact integers and half-integers
    with tf.GradientTape() as tape:
      tape.watch(x)
      y = soft_round_ops.soft_round_inverse(x, alpha=alpha)
    dy = tape.gradient(y, x)
    self.assertAllEqual(tf.math.is_finite(y), tf.ones(x.shape, dtype=bool))
    is_finite = tf.math.is_finite(dy)
    expected_finite = tf.ones(dy.shape, dtype=bool)
    if alpha > 15:
      # We allow non-finite values at 0 for large alphas, since the function
      # simply is extremely steep there.
      expected_finite = tf.tensor_scatter_nd_update(
          expected_finite, [[5]], [is_finite[5]])
    self.assertAllEqual(is_finite, expected_finite)


if __name__ == "__main__":
  tf.test.main()
