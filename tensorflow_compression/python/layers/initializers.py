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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


__all__ = [
    "IdentityInitializer",
]


class IdentityInitializer(object):
  """Initialize to the identity kernel with the given shape.

  This creates an n-D kernel suitable for `SignalConv*` with the requested
  support that produces an output identical to its input (except possibly at the
  signal boundaries).

  Note: The identity initializer in `tf.initializers` is only suitable for
  matrices, not for n-D convolution kernels (i.e., no spatial support).
  """

  def __init__(self, gain=1):
    self.gain = float(gain)

  def __call__(self, shape, dtype=None, partition_info=None):
    del partition_info  # unused
    assert len(shape) > 2, shape

    support = tuple(shape[:-2]) + (1, 1)
    indices = [[s // 2 for s in support]]
    updates = tf.constant([self.gain], dtype=dtype)
    kernel = tf.scatter_nd(indices, updates, support)

    assert shape[-2] == shape[-1], shape
    if shape[-1] != 1:
      kernel *= tf.eye(shape[-1], dtype=dtype)

    return kernel
