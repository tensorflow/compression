# Copyright 2022 Google LLC. All Rights Reserved.
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
"""Quantization tests."""

import time
from absl.testing import parameterized
import tensorflow as tf
from tensorflow_compression.python.ops import gen_ops


class QuantizationOpsTest(tf.test.TestCase, parameterized.TestCase):
  """Python test for quantization ops."""

  @parameterized.parameters(tf.bfloat16, tf.float16, tf.float32)
  def test_difference_is_at_most_one(self, dtype):
    values = tf.random.uniform((100,), -100., 100., dtype=dtype)
    rounded = gen_ops.stochastic_round(values, 1., ())
    self.assertEqual(rounded.dtype, tf.int32)
    self.assertAllClose(values, rounded, atol=1, rtol=0)

  def test_identical_seed_yields_identical_output(self):
    values = tf.random.uniform((100,), -100., 100., dtype=tf.float32)
    rounded1 = gen_ops.stochastic_round(values, 1., (123, 456))
    self.assertEqual(rounded1.dtype, tf.int32)
    rounded2 = gen_ops.stochastic_round(values, 1., (123, 456))
    self.assertEqual(rounded2.dtype, tf.int32)
    rounded3 = gen_ops.stochastic_round(values, 1., (456, 789))
    self.assertEqual(rounded3.dtype, tf.int32)
    self.assertAllEqual(rounded1, rounded2)
    self.assertNotAllEqual(rounded1, rounded3)

  def test_clock_seed_yields_different_output(self):
    values = tf.random.uniform((100,), -100., 100., dtype=tf.float32)
    rounded1 = gen_ops.stochastic_round(values, 1., ())
    self.assertEqual(rounded1.dtype, tf.int32)
    time.sleep(1.)  # Ensure even on a low-resolution clock, we change seed.
    rounded2 = gen_ops.stochastic_round(values, 1., ())
    self.assertEqual(rounded2.dtype, tf.int32)
    self.assertNotAllEqual(rounded1, rounded2)

  @parameterized.parameters(1., .75, 1e-4)
  def test_rounding_is_deterministic_at_integers(self, step_size):
    values = tf.random.uniform((100,), -100, 100, dtype=tf.int32)
    rounded = gen_ops.stochastic_round(
        step_size * tf.cast(values, tf.float32), step_size, ())
    self.assertEqual(rounded.dtype, tf.int32)
    self.assertAllEqual(values, rounded)

  @parameterized.parameters(1., .75, 1e-4)
  def test_difference_at_half_integers_is_at_most_one_half(self, step_size):
    values = tf.range(-10, 10, dtype=tf.float32) + .5
    rounded = gen_ops.stochastic_round(step_size * values, step_size, ())
    self.assertEqual(rounded.dtype, tf.int32)
    self.assertAllClose(values, rounded, atol=.5, rtol=0)

  def test_rounding_is_unbiased(self):
    values = tf.random.uniform((20,), -100., 100., dtype=tf.float32)
    replicated = tf.broadcast_to(values, (100000, 20))
    rounded = gen_ops.stochastic_round(replicated, 1., ())
    self.assertEqual(rounded.dtype, tf.int32)
    averaged = tf.reduce_mean(tf.cast(rounded, tf.float32), axis=0)
    self.assertAllClose(values, averaged, atol=5e-3, rtol=0)


if __name__ == "__main__":
  tf.test.main()
