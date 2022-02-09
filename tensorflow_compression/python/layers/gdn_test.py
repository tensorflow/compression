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

import os
from absl.testing import parameterized
import tensorflow as tf
from tensorflow_compression.python.layers import gdn
from tensorflow_compression.python.layers import parameters


class GDNTest(tf.test.TestCase, parameterized.TestCase):

  def test_invalid_data_format_raises_error(self):
    with self.assertRaises(ValueError):
      gdn.GDN(data_format="NHWC")

  @parameterized.parameters("channels_first", "channels_last")
  def test_vector_input_raises_error(self, data_format):
    x = tf.zeros((3,), dtype=tf.float32)
    with self.assertRaises(ValueError):
      gdn.GDN(data_format=data_format)(x)

  @parameterized.parameters(2, 3, 4, 5, 6)
  def test_channels_last_has_correct_output(self, rank):
    # This tests that the layer produces the correct output for a number of
    # different input dimensionalities with 'channels_last' data format.
    x = tf.random.uniform((1, 2, 3, 4, 5, 6)[:rank], dtype=tf.float32)
    y = gdn.GDN(inverse=False, rectify=False, data_format="channels_last")(x)
    self.assertEqual(x.shape, y.shape)
    self.assertAllClose(y, x / (1 + .1 * abs(x)), rtol=0, atol=1e-6)

  @parameterized.parameters(2, 3, 4, 5, 6)
  def test_channels_first_has_correct_output(self, rank):
    # This tests that the layer produces the correct output for a number of
    # different input dimensionalities with 'channels_first' data format.
    x = tf.random.uniform((6, 5, 4, 3, 2, 1)[:rank], dtype=tf.float32)
    y = gdn.GDN(inverse=False, rectify=False, data_format="channels_first")(x)
    self.assertEqual(x.shape, y.shape)
    self.assertAllClose(y, x / (1 + .1 * abs(x)), rtol=0, atol=1e-6)

  def test_igdn_has_correct_output(self):
    x = tf.random.uniform((1, 2, 3, 4), dtype=tf.float32)
    y = gdn.GDN(inverse=True, rectify=False)(x)
    self.assertEqual(x.shape, y.shape)
    self.assertAllClose(y, x * (1 + .1 * abs(x)), rtol=0, atol=1e-6)

  def test_rgdn_has_correct_output(self):
    x = tf.random.uniform((1, 2, 3, 4), -.5, .5, dtype=tf.float32)
    y = gdn.GDN(inverse=False, rectify=True)(x)
    self.assertEqual(x.shape, y.shape)
    x = tf.maximum(x, 0)
    self.assertAllClose(y, x / (1 + .1 * x), rtol=0, atol=1e-6)

  def test_quadratic_gdn_has_correct_output(self):
    x = tf.random.uniform((1, 2, 3, 4), -.5, .5, dtype=tf.float32)
    y = gdn.GDN(
        inverse=False, rectify=False,
        alpha_parameter=2, epsilon_parameter=.5)(x)
    self.assertEqual(x.shape, y.shape)
    self.assertAllClose(y, x / tf.sqrt(1 + .1 * (x ** 2)), rtol=0, atol=1e-6)

  def test_fixed_gdn_has_correct_output(self):
    x = tf.random.uniform((10, 3), -.5, .5, dtype=tf.float32)
    y = gdn.GDN(
        inverse=False, rectify=False,
        beta_parameter=[0, 0, 0], gamma_parameter=tf.ones((3, 3)))(x)
    self.assertEqual(x.shape, y.shape)
    expected_y = x / tf.reduce_sum(abs(x), axis=-1, keepdims=True)
    self.assertAllClose(y, expected_y, rtol=0, atol=1e-6)

  def test_variables_are_enumerated(self):
    layer = gdn.GDN()
    layer.alpha_parameter = None
    layer.epsilon_parameter = None
    layer.build((None, 5))
    self.assertLen(layer.weights, 4)
    self.assertLen(layer.trainable_weights, 4)
    weight_names = [w.name for w in layer.weights]
    self.assertSameElements(weight_names, [
        "reparam_alpha:0", "reparam_beta:0", "reparam_gamma:0",
        "reparam_epsilon:0"])

  def test_variables_are_not_enumerated_when_overridden(self):
    layer = gdn.GDN()
    layer.beta_parameter = [1]
    layer.gamma_parameter = [.1]
    layer.build((None, 1))
    self.assertEmpty(layer.weights)
    self.assertEmpty(layer.trainable_weights)

  def test_variables_trainable_state_follows_layer(self):
    layer = gdn.GDN()
    layer.trainable = False
    layer.build((None, 1))
    self.assertLen(layer.weights, 2)
    self.assertEmpty(layer.trainable_weights)

  def test_attributes_cannot_be_set_after_build(self):
    layer = gdn.GDN()
    layer.build((None, 2))
    with self.assertRaises(RuntimeError):
      layer.inverse = True
    with self.assertRaises(RuntimeError):
      layer.rectify = True
    with self.assertRaises(RuntimeError):
      layer.data_format = "channels_first"
    with self.assertRaises(RuntimeError):
      layer.alpha_parameter = 5
    with self.assertRaises(RuntimeError):
      layer.beta_parameter = tf.ones((5,))
    with self.assertRaises(RuntimeError):
      layer.gamma_parameter = tf.ones((5, 5))
    with self.assertRaises(RuntimeError):
      layer.epsilon_parameter = 1/3
    with self.assertRaises(RuntimeError):
      layer.alpha_initializer = "ones"
    with self.assertRaises(RuntimeError):
      layer.beta_initializer = tf.keras.initializers.Ones()
    with self.assertRaises(RuntimeError):
      layer.gamma_initializer = tf.keras.initializers.Ones()
    with self.assertRaises(RuntimeError):
      layer.epsilon_initializer = "ones"

  def test_variables_receive_gradients(self):
    x = tf.random.uniform((1, 2), dtype=tf.float32)
    layer = gdn.GDN(inverse=False, rectify=True)
    with tf.GradientTape() as g:
      y = layer(x)
    grads = g.gradient(y, layer.trainable_weights)
    self.assertLen(grads, 2)
    self.assertNotIn(None, grads)
    grad_shapes = [tuple(g.shape) for g in grads]
    weight_shapes = [tuple(w.shape) for w in layer.trainable_weights]
    self.assertSameElements(grad_shapes, weight_shapes)

  @parameterized.parameters(False, True)
  def test_can_be_saved_within_functional_model(self, build):
    inputs = tf.keras.Input(shape=(5,))
    outputs = gdn.GDN()(inputs)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    layer = model.get_layer("gdn")

    with self.subTest(name="layer_created_as_expected"):
      self.assertIsInstance(layer, gdn.GDN)
      self.assertIsInstance(layer.alpha_parameter, tf.Tensor)
      self.assertEmpty(layer.alpha_parameter.shape)
      self.assertIsInstance(layer.beta_parameter, parameters.GDNParameter)
      self.assertIsInstance(layer.gamma_parameter, parameters.GDNParameter)
      self.assertIsInstance(layer.epsilon_parameter, tf.Tensor)
      self.assertEmpty(layer.epsilon_parameter.shape)

    if build:
      x = tf.random.uniform((5, 5), dtype=tf.float32)
      y = model(x)
      weight_names = [w.name for w in model.weights]

    tempdir = self.create_tempdir()
    model_path = os.path.join(tempdir, "model")
    # This should force the model to be reconstructed via configs.
    model.save(model_path, save_traces=False)

    model = tf.keras.models.load_model(model_path)

    layer = model.get_layer("gdn")
    with self.subTest(name="layer_recreated_as_expected"):
      self.assertIsInstance(layer, gdn.GDN)
      self.assertIsInstance(layer.alpha_parameter, tf.Tensor)
      self.assertEmpty(layer.alpha_parameter.shape)
      self.assertIsInstance(layer.beta_parameter, parameters.GDNParameter)
      self.assertIsInstance(layer.gamma_parameter, parameters.GDNParameter)
      self.assertIsInstance(layer.epsilon_parameter, tf.Tensor)
      self.assertEmpty(layer.epsilon_parameter.shape)

    if build:
      with self.subTest(name="model_outputs_identical"):
        self.assertAllEqual(model(x), y)

      with self.subTest(name="model_weights_identical"):
        self.assertSameElements(weight_names, [w.name for w in model.weights])

  def test_dtypes_are_correct_with_mixed_precision(self):
    tf.keras.mixed_precision.set_global_policy("mixed_float16")
    try:
      x = tf.random.uniform((4, 3), dtype=tf.float16)
      layer = gdn.GDN()
      y = layer(x)
      for variable in layer.variables:
        self.assertEqual(variable.dtype, tf.float32)
      self.assertEqual(y.dtype, tf.float16)
    finally:
      tf.keras.mixed_precision.set_global_policy(None)


if __name__ == "__main__":
  tf.test.main()
