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

from absl.testing import parameterized
import tensorflow as tf
from tensorflow_compression.python.ops import spectral_ops


class SpectralOpsTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.parameters([4], [8, 9], [5, 3, 1])
  def test_irdft_matrix_is_orthonormal(self, *shape):
    matrix = spectral_ops.irdft_matrix(shape)
    result = tf.matmul(matrix, tf.transpose(matrix))
    self.assertAllClose(result, tf.eye(tf.TensorShape(shape).num_elements()))


if __name__ == "__main__":
  tf.test.main()
