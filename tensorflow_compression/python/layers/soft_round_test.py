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
"""Tests for soft round layers."""

import tensorflow as tf
from tensorflow_compression.python.layers import soft_round
from tensorflow_compression.python.ops import soft_round_ops


class SoftRoundTest(tf.test.TestCase):

  def test_round_layer_rounds(self):
    layer = soft_round.Round()
    x = tf.linspace(-5.0, 5.0, num=50)
    y = layer(x)
    self.assertAllClose(y, tf.math.round(x))

  def test_soft_round_layer_soft_rounds(self):
    alpha = 5.0
    layer = soft_round.SoftRound(alpha=alpha)
    x = tf.linspace(-5.0, 5.0, num=50)
    y = layer(x)
    self.assertAllClose(y,
                        soft_round_ops.soft_round(x, alpha=alpha))

  def test_soft_round_layer_inverse_inverse_soft_rounds(self):
    alpha = 5.0
    layer = soft_round.SoftRound(alpha=alpha, inverse=True)
    x = tf.linspace(-5.0, 5.0, num=50)
    y = layer(x)
    self.assertAllClose(
        y, soft_round_ops.soft_round_inverse(x, alpha=alpha))

  def test_conditional_mean_takes_conditional_mean(self):
    alpha = 5.0
    layer = soft_round.SoftRoundConditionalMean(alpha=alpha)
    x = tf.linspace(-5.0, 5.0, num=50)
    y = layer(x)
    self.assertAllClose(
        y, soft_round_ops.soft_round_conditional_mean(x, alpha=alpha))


if __name__ == "__main__":
  tf.test.main()
