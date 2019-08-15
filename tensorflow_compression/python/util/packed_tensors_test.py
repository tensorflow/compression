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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow.compat.v1 as tf

from tensorflow_compression.python.util import packed_tensors


class PackedTensorsTest(tf.test.TestCase):

  def test_pack_unpack(self):
    """Tests packing and unpacking tensors."""
    string = np.array(["xyz".encode("ascii")], dtype=object)
    shape = np.array([1, 3], dtype=np.int32)
    arrays = [string, shape]

    string_t = tf.placeholder(tf.string, [1])
    shape_t = tf.placeholder(tf.int32, [2])
    tensors = [string_t, shape_t]

    packed = packed_tensors.PackedTensors()
    packed.pack(tensors, arrays)
    packed = packed_tensors.PackedTensors(packed.string)
    string_u, shape_u = packed.unpack(tensors)

    self.assertAllEqual(string_u, string)
    self.assertAllEqual(shape_u, shape)

  def test_model(self):
    """Tests setting and getting model."""
    packed = packed_tensors.PackedTensors()
    packed.model = "xyz"
    packed = packed_tensors.PackedTensors(packed.string)
    self.assertEqual(packed.model, "xyz")
    del packed.model


if __name__ == "__main__":
  tf.test.main()
