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
"""Tests of Y4MDataset class."""

import tensorflow as tf
from tensorflow_compression.python.datasets import y4m_dataset


def shaped_uint8(string, shape):
  vector = tf.constant([int(c) for c in string], dtype=tf.uint8)
  return tf.reshape(vector, shape)


class Y4MDatasetTest(tf.test.TestCase):

  def setUp(self):
    super().setUp()
    self.tempfile_1 = self.create_tempfile(
        content=b"YUV4MPEG2 W4 H2 F30:1 Ip A0:0 C420jpeg\nFRAME\nABCDEFGHIJKL")
    self.tempfile_2 = self.create_tempfile(
        content=b"YUV4MPEG2 C444 W1 H1\nFRAME\nabcFRAME\ndef")

  def test_dataset_yields_correct_sequence(self):
    ds = y4m_dataset.Y4MDataset(
        [self.tempfile_1.full_path, self.tempfile_2.full_path])
    it = iter(ds)

    y, cbcr = next(it)
    self.assertEqual(tf.uint8, y.dtype)
    self.assertEqual(tf.uint8, cbcr.dtype)
    cb, cr = tf.unstack(cbcr, axis=-1)
    self.assertAllEqual(shaped_uint8(b"ABCDEFGH", (2, 4, 1)), y)
    self.assertAllEqual(shaped_uint8(b"IJ", (1, 2)), cb)
    self.assertAllEqual(shaped_uint8(b"KL", (1, 2)), cr)

    y, cbcr = next(it)
    self.assertEqual(tf.uint8, y.dtype)
    self.assertEqual(tf.uint8, cbcr.dtype)
    self.assertAllEqual(shaped_uint8(b"a", (1, 1, 1)), y)
    self.assertAllEqual(shaped_uint8(b"bc", (1, 1, 2)), cbcr)

    y, cbcr = next(it)
    self.assertEqual(tf.uint8, y.dtype)
    self.assertEqual(tf.uint8, cbcr.dtype)
    self.assertAllEqual(shaped_uint8(b"d", (1, 1, 1)), y)
    self.assertAllEqual(shaped_uint8(b"ef", (1, 1, 2)), cbcr)

    with self.assertRaises(StopIteration):
      next(it)


if __name__ == "__main__":
  tf.test.main()
