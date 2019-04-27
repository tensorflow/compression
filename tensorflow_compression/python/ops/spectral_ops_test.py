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
"""Tests of spectral_ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from tensorflow_compression.python.ops import spectral_ops


class SpectralOpsTest(tf.test.TestCase):

  def test_irdft1_matrix(self):
    for shape in [(4,), (3,)]:
      size = shape[0]
      matrix = spectral_ops.irdft_matrix(shape)
      # Test that the matrix is orthonormal.
      result = tf.matmul(matrix, tf.transpose(matrix))
      with self.test_session() as sess:
        result, = sess.run([result])
        self.assertAllClose(result, np.identity(size))

  def test_irdft2_matrix(self):
    for shape in [(7, 4), (8, 9)]:
      size = shape[0] * shape[1]
      matrix = spectral_ops.irdft_matrix(shape)
      # Test that the matrix is orthonormal.
      result = tf.matmul(matrix, tf.transpose(matrix))
      with self.test_session() as sess:
        result, = sess.run([result])
        self.assertAllClose(result, np.identity(size))

  def test_irdft3_matrix(self):
    for shape in [(3, 4, 2), (6, 3, 1)]:
      size = shape[0] * shape[1] * shape[2]
      matrix = spectral_ops.irdft_matrix(shape)
      # Test that the matrix is orthonormal.
      result = tf.matmul(matrix, tf.transpose(matrix))
      with self.test_session() as sess:
        result, = sess.run([result])
        self.assertAllClose(result, np.identity(size))


if __name__ == "__main__":
  tf.test.main()
