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
"""Tests of GDN layer."""

import tensorflow.compat.v2 as tf

from tensorflow_compression.python.layers import gdn


class GDNTest(tf.test.TestCase):

  def test_invalid_data_format_raises_error(self):
    x = tf.random.uniform((1, 2, 3, 4), dtype=tf.float32)
    with self.assertRaises(ValueError):
      gdn.GDN(inverse=False, rectify=False, data_format="NHWC")(x)

  def test_vector_input_raises_error(self):
    x = tf.random.uniform((3,), dtype=tf.float32)
    with self.assertRaises(ValueError):
      gdn.GDN(inverse=False, rectify=False, data_format="channels_last")(x)
    with self.assertRaises(ValueError):
      gdn.GDN(inverse=True, rectify=True, data_format="channels_first")(x)

  def test_channels_last_has_correct_output(self):
    # This tests that the layer produces the correct output for a number of
    # different input dimensionalities with 'channels_last' data format.
    for ndim in [2, 3, 4, 5, 6]:
      x = tf.random.uniform((1, 2, 3, 4, 5, 6)[:ndim], dtype=tf.float32)
      y = gdn.GDN(inverse=False, rectify=False, data_format="channels_last")(x)
      self.assertEqual(x.shape, y.shape)
      self.assertAllClose(y, x / tf.sqrt(1 + .1 * (x ** 2)), rtol=0, atol=1e-6)

  def test_channels_first_has_correct_output(self):
    # This tests that the layer produces the correct output for a number of
    # different input dimensionalities with 'channels_first' data format.
    for ndim in [2, 3, 4, 5, 6]:
      x = tf.random.uniform((6, 5, 4, 3, 2, 1)[:ndim], dtype=tf.float32)
      y = gdn.GDN(inverse=False, rectify=False, data_format="channels_first")(x)
      self.assertEqual(x.shape, y.shape)
      self.assertAllClose(y, x / tf.sqrt(1 + .1 * (x ** 2)), rtol=0, atol=1e-6)

  def test_igdn_has_correct_output(self):
    x = tf.random.uniform((1, 2, 3, 4), dtype=tf.float32)
    y = gdn.GDN(inverse=True, rectify=False)(x)
    self.assertEqual(x.shape, y.shape)
    self.assertAllClose(y, x * tf.sqrt(1 + .1 * (x ** 2)), rtol=0, atol=1e-6)

  def test_rgdn_has_correct_output(self):
    x = tf.random.uniform((1, 2, 3, 4), -.5, .5, dtype=tf.float32)
    y = gdn.GDN(inverse=False, rectify=True)(x)
    self.assertEqual(x.shape, y.shape)
    x = tf.maximum(x, 0)
    self.assertAllClose(y, x / tf.sqrt(1 + .1 * (x ** 2)), rtol=0, atol=1e-6)

  def test_variables_receive_gradients(self):
    x = tf.random.uniform((1, 2), dtype=tf.float32)
    layer = gdn.GDN(inverse=False, rectify=True)
    with tf.GradientTape() as g:
      y = layer(x)
    grads = g.gradient(y, layer.trainable_variables)
    self.assertLen(grads, 2)
    self.assertNotIn(None, grads)


if __name__ == "__main__":
  tf.test.main()
