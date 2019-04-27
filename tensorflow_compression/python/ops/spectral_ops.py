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
"""Parameterizations for layer classes."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from scipy import fftpack
import tensorflow as tf


_matrix_cache = {}


__all__ = [
    "irdft_matrix",
]


def irdft_matrix(shape, dtype=tf.float32):
  """Matrix for implementing kernel reparameterization with `tf.matmul`.

  This can be used to represent a kernel with the provided shape in the RDFT
  domain.

  Example code for kernel creation, assuming 2D kernels:

  ```
  def create_kernel(init):
    shape = init.shape.as_list()
    matrix = irdft_matrix(shape[:2])
    init = tf.reshape(init, (shape[0] * shape[1], shape[2] * shape[3]))
    init = tf.matmul(tf.transpose(matrix), init)
    kernel = tf.Variable(init)
    kernel = tf.matmul(matrix, kernel)
    kernel = tf.reshape(kernel, shape)
    return kernel
  ```

  Args:
    shape: Iterable of integers. Shape of kernel to apply this matrix to.
    dtype: `dtype` of returned matrix.

  Returns:
    `Tensor` of shape `(prod(shape), prod(shape))` and dtype `dtype`.
  """
  shape = tuple(int(s) for s in shape)
  dtype = tf.as_dtype(dtype)
  key = (tf.get_default_graph(), "irdft", shape, dtype.as_datatype_enum)
  matrix = _matrix_cache.get(key)
  if matrix is None:
    size = np.prod(shape)
    rank = len(shape)
    matrix = np.identity(size, dtype=np.float64).reshape((size,) + shape)
    for axis in range(rank):
      matrix = fftpack.rfft(matrix, axis=axis + 1)
      slices = (rank + 1) * [slice(None)]
      if shape[axis] % 2 == 1:
        slices[axis + 1] = slice(1, None)
      else:
        slices[axis + 1] = slice(1, -1)
      matrix[tuple(slices)] *= np.sqrt(2)
    matrix /= np.sqrt(size)
    matrix = np.reshape(matrix, (size, size))
    matrix = tf.constant(
        matrix, dtype=dtype, name="irdft_" + "x".join([str(s) for s in shape]))
    _matrix_cache[key] = matrix
  return matrix
