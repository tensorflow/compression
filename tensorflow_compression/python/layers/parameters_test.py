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
"""Tests of parameters."""

import tensorflow as tf
from tensorflow_compression.python.layers import parameters


class ParameterTest:

  def test_initial_value_is_reproduced(self):
    initial_value = tf.random.uniform(self.shape, dtype=tf.float32)
    parameter = self.cls(initial_value, **self.kwargs)
    self.assertAllClose(initial_value, parameter(), atol=1e-6, rtol=0)

  def test_name_and_value_are_reproduced_after_serialization(self):
    initial_value = tf.random.uniform(self.shape, dtype=tf.float32)
    parameter = self.cls(initial_value, **self.kwargs)
    name_before = parameter.name
    value_before = parameter()
    json = tf.keras.utils.serialize_keras_object(parameter)
    weights = parameter.get_weights()
    parameter = tf.keras.utils.deserialize_keras_object(json)
    self.assertIsInstance(parameter, self.cls)
    self.assertEqual(name_before, parameter.name)
    parameter.set_weights(weights)
    value_after = parameter()
    self.assertAllEqual(value_before, value_after)
    self.assertEqual(value_before.dtype.name, value_after.dtype.name)

  def test_converts_to_tensor(self):
    initial_value = tf.random.uniform(self.shape, dtype=tf.float32)
    parameter = self.cls(initial_value, **self.kwargs)
    value = parameter()
    converted = tf.convert_to_tensor(parameter)
    self.assertAllEqual(value, converted)
    self.assertEqual(value.dtype.name, converted.dtype.name)


class RDFTParameterTest(ParameterTest, tf.test.TestCase):

  cls = parameters.RDFTParameter
  kwargs = dict(name="hello_rdft", dc=True)
  shape = (3, 3, 1, 2)

  def test_initial_value_is_reproduced_without_dc(self):
    initial_value = tf.random.uniform(self.shape, dtype=tf.float32)
    parameter = self.cls(initial_value, dc=False)
    expected_value = initial_value - tf.reduce_mean(
        initial_value, axis=(0, 1), keepdims=True)
    self.assertAllClose(expected_value, parameter(), atol=1e-6, rtol=0)


class GDNParameterTest(ParameterTest, tf.test.TestCase):

  cls = parameters.GDNParameter
  kwargs = dict(name="hello_gdn")
  shape = (2, 1, 3)

  def test_initial_value_is_reproduced_with_minimum(self):
    initial_value = tf.random.uniform(self.shape, dtype=tf.float32)
    parameter = self.cls(initial_value, minimum=.5)
    expected_value = tf.maximum(initial_value, .5)
    self.assertAllClose(expected_value, parameter(), atol=1e-6, rtol=0)


if __name__ == "__main__":
  tf.test.main()
