# Copyright 2021 Google LLC. All Rights Reserved.
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
"""Tests of initializers."""

import tensorflow as tf
from tensorflow_compression.python.layers import initializers


class InitializerTest(tf.test.TestCase):

  def test_creates_1d_kernel(self):
    expected_kernel = tf.transpose([
        [[0, 3, 0], [0, 0, 0], [0, 0, 0]],
        [[0, 0, 0], [0, 3, 0], [0, 0, 0]],
        [[0, 0, 0], [0, 0, 0], [0, 3, 0]],
        [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
    ], (2, 0, 1))
    kernel = initializers.IdentityInitializer(gain=3)((3, 4, 3), dtype=tf.int32)
    self.assertAllEqual(expected_kernel, kernel)

  def test_creates_2d_kernel(self):
    expected_kernel = tf.constant([
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0],
    ])[:, :, None, None]
    kernel = initializers.IdentityInitializer()((4, 5, 1, 1), dtype=tf.float32)
    self.assertAllEqual(expected_kernel, kernel)

  def test_fails_for_invalid_shape(self):
    with self.assertRaises(ValueError):
      initializers.IdentityInitializer()((2, 3), dtype=tf.float32)


if __name__ == "__main__":
  tf.test.main()
