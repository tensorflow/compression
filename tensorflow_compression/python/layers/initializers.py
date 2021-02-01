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
"""Initializers for layer classes."""

import tensorflow as tf


__all__ = [
    "IdentityInitializer",
]


class IdentityInitializer(tf.keras.initializers.Initializer):
  """Initialize to the identity kernel with the given shape.

  This creates an n-D kernel suitable for `SignalConv*` with the requested
  support that produces an output identical to its input (except possibly at the
  signal boundaries).

  Note: The identity initializer in `tf.keras.initializers` is only suitable for
  matrices, not for n-D convolution kernels (i.e., no spatial support).
  """

  def __init__(self, gain=1):
    super().__init__()
    self.gain = gain

  def __call__(self, shape, dtype=None, **kwargs):
    del kwargs  # unused
    shape = tf.TensorShape(shape)
    if shape.rank <= 2:
      raise ValueError(f"shape must be at least rank 3, got {shape}.")

    support = shape[:-2] + (1, 1)
    indices = [[s // 2 for s in support]]
    updates = tf.constant([self.gain], dtype=dtype)
    spatial_kernel = tf.scatter_nd(indices, updates, support)
    return spatial_kernel * tf.eye(shape[-2], shape[-1], dtype=dtype)

  def get_config(self):
    config = super().get_config()
    config.update(gain=self.gain)
    return config
