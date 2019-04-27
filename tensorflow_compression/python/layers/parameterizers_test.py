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
"""Tests of parameterizers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from tensorflow_compression.python.layers import parameterizers


class ParameterizersTest(tf.test.TestCase):

  def _test_parameterizer(self, param, init, shape):
    var = param(
        getter=tf.get_variable, name="test", shape=shape, dtype=tf.float32,
        initializer=init, regularizer=None)
    with self.test_session() as sess:
      tf.global_variables_initializer().run()
      var, = sess.run([var])
    return var

  def test_static_parameterizer(self):
    shape = (1, 2, 3, 4)
    var = self._test_parameterizer(
        parameterizers.StaticParameterizer(tf.initializers.zeros()),
        tf.initializers.random_uniform(), shape)
    self.assertEqual(var.shape, shape)
    self.assertAllClose(var, np.zeros(shape), rtol=0, atol=1e-7)

  def test_rdft_parameterizer(self):
    shape = (3, 4, 2, 1)
    var = self._test_parameterizer(
        parameterizers.RDFTParameterizer(),
        tf.initializers.ones(), shape)
    self.assertEqual(var.shape, shape)
    self.assertAllClose(var, np.ones(shape), rtol=0, atol=1e-6)

  def test_nonnegative_parameterizer(self):
    shape = (1, 2, 3, 4)
    var = self._test_parameterizer(
        parameterizers.NonnegativeParameterizer(),
        tf.initializers.random_uniform(), shape)
    self.assertEqual(var.shape, shape)
    self.assertTrue(np.all(var >= 0))

  def test_positive_parameterizer(self):
    shape = (1, 2, 3, 4)
    var = self._test_parameterizer(
        parameterizers.NonnegativeParameterizer(minimum=.1),
        tf.initializers.random_uniform(), shape)
    self.assertEqual(var.shape, shape)
    self.assertTrue(np.all(var >= .1))


if __name__ == "__main__":
  tf.test.main()
