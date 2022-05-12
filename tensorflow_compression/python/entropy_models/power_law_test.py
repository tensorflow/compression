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
"""Tests of power law entropy model."""

import numpy as np
import tensorflow as tf
from tensorflow_compression.python.entropy_models.power_law import PowerLawEntropyModel


class PowerLawEntropyModelTest(tf.test.TestCase):

  def test_can_instantiate(self):
    em = PowerLawEntropyModel(coding_rank=1)
    self.assertEqual(em.coding_rank, 1)
    self.assertEqual(em.bottleneck_dtype, tf.float32)

  def test_requires_coding_rank_greater_equal_zero(self):
    with self.assertRaises(ValueError):
      PowerLawEntropyModel(coding_rank=-1)

  def test_quantizes_to_integers(self):
    em = PowerLawEntropyModel(coding_rank=1)
    x = tf.range(-20., 20.)
    x_perturbed = x + tf.random.uniform(x.shape, -.49, .49)
    x_quantized = em.quantize(x_perturbed)
    self.assertAllEqual(x, x_quantized)

  def test_gradients_are_straight_through(self):
    em = PowerLawEntropyModel(coding_rank=1)
    x = tf.range(-20., 20.)
    x_perturbed = x + tf.random.uniform(x.shape, -.49, .49)
    with tf.GradientTape() as tape:
      tape.watch(x_perturbed)
      x_quantized = em.quantize(x_perturbed)
    gradients = tape.gradient(x_quantized, x_perturbed)
    self.assertAllEqual(gradients, tf.ones_like(gradients))

  def test_compression_consistent_with_quantization(self):
    em = PowerLawEntropyModel(coding_rank=1)
    x = tf.range(-20., 20.)
    x += tf.random.uniform(x.shape, -.49, .49)
    x_quantized = em.quantize(x)
    x_decompressed = em.decompress(em.compress(x), x.shape)
    self.assertAllEqual(x_decompressed, x_quantized)

  def test_penalty_is_proportional_to_code_length(self):
    em = PowerLawEntropyModel(coding_rank=1)
    x = tf.range(-20., 20.)[:, None]
    x += tf.random.uniform(x.shape, -.49, .49)
    strings = em.compress(tf.broadcast_to(x, (40, 100)))
    code_lengths = tf.cast(tf.strings.length(strings, unit="BYTE"), tf.float32)
    code_lengths *= 8 / 100
    penalties = em.penalty(x)
    # There are some fluctuations due to `alpha`, `floor`, and rounding, but we
    # expect a high degree of correlation between code lengths and penalty.
    self.assertGreater(np.corrcoef(code_lengths, penalties)[0, 1], .96)

  def test_penalty_is_nonnegative_and_differentiable(self):
    em = PowerLawEntropyModel(coding_rank=1)
    x = tf.range(-20., 20.)[:, None]
    x += tf.random.uniform(x.shape, -.49, .49)
    with tf.GradientTape() as tape:
      tape.watch(x)
      penalties = em.penalty(x)
    gradients = tape.gradient(penalties, x)
    self.assertAllGreaterEqual(penalties, 0)
    self.assertAllEqual(tf.sign(gradients), tf.sign(x))

  def test_compression_works_in_tf_function(self):
    samples = tf.random.stateless_normal([100], (34, 232))

    # Since tf.function traces each function twice, and only allows variable
    # creation in the first call, we need to have a stateful object in which we
    # create the entropy model only the first time the function is called, and
    # store it for the second time.

    class Compressor:

      def compress(self, values):
        if not hasattr(self, "em"):
          self.em = PowerLawEntropyModel(coding_rank=1)
        compressed = self.em.compress(values)
        return self.em.decompress(compressed, [100])

    values_eager = Compressor().compress(samples)
    values_function = tf.function(Compressor().compress)(samples)
    self.assertAllClose(samples, values_eager, rtol=0., atol=.5)
    self.assertAllEqual(values_eager, values_function)

  def test_dtypes_are_correct_with_mixed_precision(self):
    tf.keras.mixed_precision.set_global_policy("mixed_float16")
    try:
      em = PowerLawEntropyModel(coding_rank=1)
      self.assertEqual(em.bottleneck_dtype, tf.float16)
      x = tf.random.stateless_normal((2, 5), seed=(0, 1), dtype=tf.float16)
      x_tilde, penalty = em(x)
      bitstring = em.compress(x)
      x_hat = em.decompress(bitstring, (5,))
      self.assertEqual(x_hat.dtype, tf.float16)
      self.assertAllClose(x, x_hat, rtol=0, atol=.5)
      self.assertEqual(x_tilde.dtype, tf.float16)
      self.assertAllClose(x, x_tilde, rtol=0, atol=.5)
      self.assertEqual(penalty.dtype, tf.float16)
      self.assertEqual(penalty.shape, (2,))
    finally:
      tf.keras.mixed_precision.set_global_policy(None)


if __name__ == "__main__":
  tf.test.main()
