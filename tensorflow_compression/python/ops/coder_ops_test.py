# -*- coding: utf-8 -*-
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
"""Coder operations tests."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports

import tensorflow as tf

from tensorflow.python.platform import test

import tensorflow_compression as tfc


class CoderOpsTest(test.TestCase):
  """Coder ops test.

  Coder ops have C++ tests. Python test just ensures that Python binding is not
  broken.
  """

  def testReadmeExample(self):
    data = tf.random_uniform((128, 128), 0, 10, dtype=tf.int32)
    histogram = tf.bincount(data, minlength=10, maxlength=10)
    cdf = tf.cumsum(histogram, exclusive=False)
    cdf = tf.pad(cdf, [[1, 0]])
    cdf = tf.reshape(cdf, [1, 1, -1])

    data = tf.cast(data, tf.int16)
    encoded = tfc.range_encode(data, cdf, precision=14)
    decoded = tfc.range_decode(encoded, tf.shape(data), cdf, precision=14)

    with self.test_session() as sess:
      self.assertAllEqual(*sess.run((data, decoded)))


if __name__ == "__main__":
  test.main()
