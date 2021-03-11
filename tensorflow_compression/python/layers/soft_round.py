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
"""Layers for soft rounding."""

import tensorflow as tf
from tensorflow_compression.python.ops import soft_round_ops


__all__ = [
    "Round",
    "SoftRound",
    "SoftRoundConditionalMean",
]


class Round(tf.keras.layers.Layer):
  """Applies rounding."""

  def call(self, inputs):
    return tf.round(inputs)

  def compute_output_shape(self, input_shape):
    return input_shape


class SoftRound(tf.keras.layers.Layer):
  """Applies a differentiable approximation of rounding."""

  def __init__(self,
               alpha=5.0,
               inverse=False,
               **kwargs):
    super().__init__(**kwargs)
    self._alpha = alpha
    self._transform = (
        soft_round_ops.soft_round_inverse
        if inverse else soft_round_ops.soft_round)

  def call(self, inputs):
    outputs = self._transform(inputs, self._alpha)
    return outputs

  def compute_output_shape(self, input_shape):
    return input_shape


class SoftRoundConditionalMean(tf.keras.layers.Layer):
  """Conditional mean of inputs given noisy soft rounded values."""

  def __init__(self,
               alpha=5.0,
               **kwargs):
    super().__init__(**kwargs)
    self._alpha = alpha

  def call(self, inputs):
    return soft_round_ops.soft_round_conditional_mean(
        inputs, alpha=self._alpha)

  def compute_output_shape(self, input_shape):
    return input_shape
