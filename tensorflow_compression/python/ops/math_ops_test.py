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
"""Tests for the math operations."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow_compression.python.ops import math_ops


class MathTest(tf.test.TestCase):

  def _test_upper_bound(self, gradient):
    inputs = tf.placeholder(dtype=tf.float32)
    outputs = math_ops.upper_bound(inputs, 0, gradient=gradient)
    pgrads, = tf.gradients([outputs], [inputs], [tf.ones_like(inputs)])
    ngrads, = tf.gradients([outputs], [inputs], [-tf.ones_like(inputs)])

    inputs_feed = [-1, 1]
    outputs_expected = [-1, 0]
    if gradient == "disconnected":
      pgrads_expected = [1, 0]
      ngrads_expected = [-1, 0]
    elif gradient == "identity":
      pgrads_expected = [1, 1]
      ngrads_expected = [-1, -1]
    else:
      pgrads_expected = [1, 1]
      ngrads_expected = [-1, 0]

    with self.test_session() as sess:
      outputs, pgrads, ngrads = sess.run(
          [outputs, pgrads, ngrads], {inputs: inputs_feed})
      self.assertAllEqual(outputs, outputs_expected)
      self.assertAllEqual(pgrads, pgrads_expected)
      self.assertAllEqual(ngrads, ngrads_expected)

  def test_upper_bound_disconnected(self):
    self._test_upper_bound("disconnected")

  def test_upper_bound_identity(self):
    self._test_upper_bound("identity")

  def test_upper_bound_identity_if_towards(self):
    self._test_upper_bound("identity_if_towards")

  def test_upper_bound_invalid(self):
    with self.assertRaises(ValueError):
      self._test_upper_bound("invalid")

  def _test_lower_bound(self, gradient):
    inputs = tf.placeholder(dtype=tf.float32)
    outputs = math_ops.lower_bound(inputs, 0, gradient=gradient)
    pgrads, = tf.gradients([outputs], [inputs], [tf.ones_like(inputs)])
    ngrads, = tf.gradients([outputs], [inputs], [-tf.ones_like(inputs)])

    inputs_feed = [-1, 1]
    outputs_expected = [0, 1]
    if gradient == "disconnected":
      pgrads_expected = [0, 1]
      ngrads_expected = [0, -1]
    elif gradient == "identity":
      pgrads_expected = [1, 1]
      ngrads_expected = [-1, -1]
    else:
      pgrads_expected = [0, 1]
      ngrads_expected = [-1, -1]

    with self.test_session() as sess:
      outputs, pgrads, ngrads = sess.run(
          [outputs, pgrads, ngrads], {inputs: inputs_feed})
      self.assertAllEqual(outputs, outputs_expected)
      self.assertAllEqual(pgrads, pgrads_expected)
      self.assertAllEqual(ngrads, ngrads_expected)

  def test_lower_bound_disconnected(self):
    self._test_lower_bound("disconnected")

  def test_lower_bound_identity(self):
    self._test_lower_bound("identity")

  def test_lower_bound_identity_if_towards(self):
    self._test_lower_bound("identity_if_towards")

  def test_lower_bound_invalid(self):
    with self.assertRaises(ValueError):
      self._test_lower_bound("invalid")


if __name__ == "__main__":
  tf.test.main()
