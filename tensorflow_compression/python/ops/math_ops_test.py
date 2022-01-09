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
"""Tests for the math operations."""

from absl.testing import parameterized
import scipy.stats
import tensorflow as tf
from tensorflow_compression.python.ops import math_ops
from tensorflow_compression.python.ops import round_ops


class MathTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.parameters("disconnected", "identity", "identity_if_towards")
  def test_upper_bound_has_correct_outputs_and_gradients(self, gradient):
    inputs = tf.constant([-1, 1], dtype=tf.float32)
    with tf.GradientTape(persistent=True) as tape:
      tape.watch(inputs)
      outputs = math_ops.upper_bound(inputs, 0, gradient=gradient)
    pgrads = tape.gradient(outputs, inputs, tf.ones_like(inputs))
    ngrads = tape.gradient(outputs, inputs, -tf.ones_like(inputs))
    self.assertAllEqual(outputs, [-1, 0])
    if gradient == "disconnected":
      self.assertAllEqual(pgrads, [1, 0])
      self.assertAllEqual(ngrads, [-1, 0])
    elif gradient == "identity":
      self.assertAllEqual(pgrads, [1, 1])
      self.assertAllEqual(ngrads, [-1, -1])
    else:
      self.assertAllEqual(pgrads, [1, 1])
      self.assertAllEqual(ngrads, [-1, 0])

  def test_upper_bound_invalid(self):
    with self.assertRaises(ValueError):
      math_ops.upper_bound(tf.zeros((1, 2)), 0, gradient="invalid")

  @parameterized.parameters("disconnected", "identity", "identity_if_towards")
  def test_lower_bound_has_correct_outputs_and_gradients(self, gradient):
    inputs = tf.constant([-1, 1], dtype=tf.float32)
    with tf.GradientTape(persistent=True) as tape:
      tape.watch(inputs)
      outputs = math_ops.lower_bound(inputs, 0, gradient=gradient)
    pgrads = tape.gradient(outputs, inputs, tf.ones_like(inputs))
    ngrads = tape.gradient(outputs, inputs, -tf.ones_like(inputs))
    self.assertAllEqual(outputs, [0, 1])
    if gradient == "disconnected":
      self.assertAllEqual(pgrads, [0, 1])
      self.assertAllEqual(ngrads, [0, -1])
    elif gradient == "identity":
      self.assertAllEqual(pgrads, [1, 1])
      self.assertAllEqual(ngrads, [-1, -1])
    else:
      self.assertAllEqual(pgrads, [0, 1])
      self.assertAllEqual(ngrads, [-1, -1])

  def test_lower_bound_invalid(self):
    with self.assertRaises(ValueError):
      math_ops.lower_bound(tf.zeros((1, 2)), 0, gradient="invalid")


class PerturbAndApplyTest(tf.test.TestCase):

  def test_perturb_and_apply_noise(self):
    x = tf.random.normal([10000], seed=0)
    y, x_plus_u0 = math_ops.perturb_and_apply(
        tf.identity, x, expected_grads=True)
    u0 = x_plus_u0-x
    u1 = y - x
    # Check if residuals are as expected
    self.assertAllClose(u0, u1)
    # Check if noise has expected uniform distribution
    _, p = scipy.stats.kstest(u0, "uniform", (-0.5, 1.0))
    self.assertAllLessEqual(tf.abs(u0), 0.5)
    self.assertGreater(p, 1e-6)

  def test_perturb_and_apply_gradient_soft_round(self):
    f = round_ops.soft_round
    x = tf.linspace(-2.0, 2.0, 200)
    temperature = 7.0
    with tf.GradientTape(persistent=True) as g:
      g.watch(x)
      y = math_ops.perturb_and_apply(f, x, temperature, expected_grads=True)[0]
    dx = g.gradient(y, x)
    self.assertAllClose(dx, tf.ones_like(dx))

  def test_perturb_and_apply_gradient_parabola(self):
    f = lambda x, a: a*x*x
    x = tf.linspace(-2.0, 2.0, 200)
    a = 7.0
    with tf.GradientTape(persistent=True) as g:
      g.watch(x)
      y = math_ops.perturb_and_apply(f, x, a, expected_grads=True)[0]
    dx = g.gradient(y, x)
    self.assertAllClose(dx, f(x+.5, a)-f(x-.5, a))


if __name__ == "__main__":
  tf.test.main()
