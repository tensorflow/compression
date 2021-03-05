# Copyright 2019 Google LLC. All Rights Reserved.
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
"""Tests of PackedTensors class."""

import tensorflow as tf
from tensorflow_compression.python.util import packed_tensors


class PackedTensorsTest(tf.test.TestCase):

  def test_pack_unpack_identity(self):
    """Tests packing and unpacking tensors returns the same values."""
    string = tf.constant(["xyz"], dtype=tf.string)
    shape = tf.constant([1, 3], dtype=tf.int32)
    packed = packed_tensors.PackedTensors()
    packed.pack([string, shape])
    packed = packed_tensors.PackedTensors(packed.string)
    string_unpacked, shape_unpacked = packed.unpack([tf.string, tf.int32])
    self.assertAllEqual(string_unpacked, string)
    self.assertAllEqual(shape_unpacked, shape)

  def test_set_get_model_identity(self):
    """Tests setting and getting model returns the same value."""
    packed = packed_tensors.PackedTensors()
    packed.model = "xyz"
    packed = packed_tensors.PackedTensors(packed.string)
    self.assertEqual(packed.model, "xyz")
    del packed.model


if __name__ == "__main__":
  tf.test.main()
