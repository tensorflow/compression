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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from tensorflow_compression.python.layers import gdn


class GDNTest(tf.test.TestCase):

  def _run_gdn(self, x, shape, inverse, rectify, data_format):
    inputs = tf.placeholder(tf.float32, shape)
    layer = gdn.GDN(
        inverse=inverse, rectify=rectify, data_format=data_format)
    outputs = layer(inputs)
    with self.test_session() as sess:
      tf.global_variables_initializer().run()
      y, = sess.run([outputs], {inputs: x})
    return y

  def test_invalid_data_format(self):
    x = np.random.uniform(size=(1, 2, 3, 4))
    with self.assertRaises(ValueError):
      self._run_gdn(x, x.shape, False, False, "NHWC")

  def test_unknown_dim(self):
    x = np.random.uniform(size=(1, 2, 3, 4))
    with self.assertRaises(ValueError):
      self._run_gdn(x, 4 * [None], False, False, "channels_last")

  def test_channels_last(self):
    for ndim in [2, 3, 4, 5, 6]:
      x = np.random.uniform(size=(1, 2, 3, 4, 5, 6)[:ndim])
      y = self._run_gdn(x, x.shape, False, False, "channels_last")
      self.assertEqual(x.shape, y.shape)
      self.assertAllClose(y, x / np.sqrt(1 + .1 * (x ** 2)), rtol=0, atol=1e-6)

  def test_channels_first(self):
    for ndim in [2, 3, 4, 5, 6]:
      x = np.random.uniform(size=(6, 5, 4, 3, 2, 1)[:ndim])
      y = self._run_gdn(x, x.shape, False, False, "channels_first")
      self.assertEqual(x.shape, y.shape)
      self.assertAllClose(
          y, x / np.sqrt(1 + .1 * (x ** 2)), rtol=0, atol=1e-6)

  def test_wrong_dims(self):
    x = np.random.uniform(size=(3,))
    with self.assertRaises(ValueError):
      self._run_gdn(x, x.shape, False, False, "channels_last")
    with self.assertRaises(ValueError):
      self._run_gdn(x, x.shape, True, True, "channels_first")

  def test_igdn(self):
    x = np.random.uniform(size=(1, 2, 3, 4))
    y = self._run_gdn(x, x.shape, True, False, "channels_last")
    self.assertEqual(x.shape, y.shape)
    self.assertAllClose(y, x * np.sqrt(1 + .1 * (x ** 2)), rtol=0, atol=1e-6)

  def test_rgdn(self):
    x = np.random.uniform(-.5, .5, size=(1, 2, 3, 4))
    y = self._run_gdn(x, x.shape, False, True, "channels_last")
    self.assertEqual(x.shape, y.shape)
    x = np.maximum(x, 0)
    self.assertAllClose(y, x / np.sqrt(1 + .1 * (x ** 2)), rtol=0, atol=1e-6)


if __name__ == "__main__":
  tf.test.main()
