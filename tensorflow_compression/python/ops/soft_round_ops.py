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
"""Soft rounding ops."""

import tensorflow as tf


__all__ = ["soft_round", "soft_round_inverse", "soft_round_conditional_mean"]


def soft_round(x, alpha, eps=1e-12):
  """Differentiable approximation to round().

  Larger alphas correspond to closer approximations of the round function.
  If alpha is close to zero, this function reduces to the identity.

  This is described in Sec. 4.1. in the paper
  > "Universally Quantized Neural Compression"<br />
  > Eirikur Agustsson & Lucas Theis<br />
  > https://arxiv.org/abs/2006.09952

  Args:
    x: tf.Tensor. Inputs to the rounding function.
    alpha: Float or tf.Tensor. Controls smoothness of the approximation.
    eps: Float. Threshold below which soft_round() will return identity.

  Returns:
    tf.Tensor
  """

  if isinstance(alpha, (float, int)) and alpha < eps:
    return tf.identity(x, name="soft_round")

  m = tf.floor(x) + 0.5
  r = x - m
  z = tf.maximum(tf.tanh(alpha / 2.0) * 2.0, eps)
  y = m + tf.tanh(alpha * r) / z

  # For very low alphas, soft_round behaves like identity
  return tf.where(alpha < eps, x, y, name="soft_round")


@tf.custom_gradient
def _clip_st(s):
  """Clip s to [-1 + 1e-7, 1 - 1e-7] with straight-through gradients."""
  s = tf.clip_by_value(s, -1 + 1e-7, 1 - 1e-7)
  grad = lambda x: x
  return s, grad


def soft_round_inverse(y, alpha, eps=1e-12):
  """Inverse of soft_round().

  This is described in Sec. 4.1. in the paper
  > "Universally Quantized Neural Compression"<br />
  > Eirikur Agustsson & Lucas Theis<br />
  > https://arxiv.org/abs/2006.09952

  Args:
    y: tf.Tensor. Inputs to this function.
    alpha: Float or tf.Tensor. Controls smoothness of the approximation.
    eps: Float. Threshold below which soft_round() is assumed to equal the
      identity function.

  Returns:
    tf.Tensor
  """

  if isinstance(alpha, (float, int)) and alpha < eps:
    return tf.identity(y, name="soft_round_inverse")

  m = tf.floor(y) + 0.5
  s = (y - m) * (tf.tanh(alpha / 2.0) * 2.0)
  # We have -0.5 <= (y-m) <= 0.5 and -1 < tanh < 1, so
  # -1 <= s <= 1. However tf.atanh is only stable for inputs
  # in the range [-1+1e-7, 1-1e-7], so we (safely) clip s to this range.
  # In the rare case where `1-|s| < 1e-7`, we use straight-through for the
  # gradient.
  s = _clip_st(s)
  r = tf.atanh(s) / tf.maximum(alpha, eps)

  # For very low alphas, soft_round behaves like identity
  return tf.where(alpha < eps, y, m + r, name="soft_round_inverse")


def soft_round_conditional_mean(inputs, alpha):
  """Conditional mean of inputs given noisy soft rounded values.

  Computes g(z) = E[Y | s(Y) + U = z] where s is the soft-rounding function,
  U is uniform between -0.5 and 0.5 and `Y` is considered uniform when truncated
  to the interval [z-0.5, z+0.5].

  This is described in Sec. 4.1. in the paper
  > "Universally Quantized Neural Compression"<br />
  > Eirikur Agustsson & Lucas Theis<br />
  > https://arxiv.org/abs/2006.09952


  Args:
    inputs: The input tensor.
    alpha: The softround alpha.

  Returns:
    The conditional mean, of same shape as `inputs`.
  """
  return soft_round_inverse(inputs - 0.5, alpha) + 0.5
