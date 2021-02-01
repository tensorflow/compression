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
"""Generalized divisive normalization layer."""

from typing import Any, Callable, Dict, Union
import tensorflow as tf
from tensorflow_compression.python.layers import parameters


__all__ = [
    "GDN",
]


@tf.keras.utils.register_keras_serializable(package="tensorflow_compression")
class GDN(tf.keras.layers.Layer):
  """Generalized divisive normalization layer.

  Based on the papers:

  > "Density modeling of images using a generalized normalization
  > transformation"<br />
  > J. Ballé, V. Laparra, E.P. Simoncelli<br />
  > https://arxiv.org/abs/1511.06281

  > "End-to-end optimized image compression"<br />
  > J. Ballé, V. Laparra, E.P. Simoncelli<br />
  > https://arxiv.org/abs/1611.01704

  Implements an activation function that is essentially a multivariate
  generalization of a particular sigmoid-type function:

  ```
  y[i] = x[i] / sqrt(beta[i] + sum_j(gamma[j, i] * x[j]^2))
  ```

  where `i` and `j` run over channels. This implementation never sums across
  spatial dimensions. It is similar to local response normalization, but much
  more flexible, as `beta` and `gamma` are trainable parameters.

  Attributes:
    inverse: Boolean. If `False`, compute GDN response. If `True`, compute IGDN
      response (one step of fixed point iteration to invert GDN; the division is
      replaced by multiplication).
    rectify: Boolean. If `True`, apply a `relu` nonlinearity to the inputs
      before calculating GDN response.
    data_format: String. Format of input tensor. Currently supports
      `'channels_first'` and `'channels_last'`.
    beta_parameter: Tensor, callable, or `None`. A `tf.Tensor` means that the
      value of beta is fixed. A callable can be used to determine the value of
      beta as a function of some other variable or tensor. This can be a
      `Parameter` object. `None` means that when the layer is built, a
      `GDNParameter` object is created to train beta (with a minimum value of
      1e-6). Note that certain choices here such as `tf.Tensor`s or lambda
      functions may prevent JSON-style serialization (`Parameter` objects work).
    gamma_parameter: Tensor, callable, or `None`. A `tf.Tensor` means that the
      value of gamma is fixed. A callable can be used to determine the value of
      gamma as a function of some other variable or tensor. This can be a
      `Parameter` object. `None` means that when the layer is built, a
      `GDNParameter` object is created to train gamma. Note that certain choices
      here such as `tf.Tensor`s or lambda functions may prevent JSON-style
      serialization (`Parameter` objects work).
    beta_initializer: `Initializer` object for beta parameter. Only used if beta
      is created on building the layer. Defaults to 1.
    gamma_initializer: `Initializer` object for gamma parameter. Only used if
      gamma is created on building the layer. Defaults to identity matrix
      multiplied by 0.1. A good default value for the diagonal is somewhere
      between 0 and 0.5. If set to 0 and beta initialized as 1, the layer is
      effectively initialized to the identity operation.
    beta: `tf.Tensor`. Read-only property always returning the current value of
      beta.
    gamma: `tf.Tensor`. Read-only property always returning the current value of
      gamma.
  """

  def __init__(self,
               inverse=False,
               rectify=False,
               data_format="channels_last",
               beta_parameter=None,
               gamma_parameter=None,
               beta_initializer="ones",
               gamma_initializer=tf.keras.initializers.Identity(.1),
               **kwargs):
    """Initializer.

    Args:
      inverse: Boolean. Initial value of eponymous attribute.
      rectify: Boolean. Initial value of eponymous attribute.
      data_format: String. Initial value of eponymous attribute.
      beta_parameter: Tensor, callable, or `None`. Initial value of eponymous
        attribute.
      gamma_parameter: Tensor, callable, or `None`. Initial value of eponymous
        attribute.
      beta_initializer: `Initializer` object. Initial value of eponymous
        attribute.
      gamma_initializer: `Initializer` object. Initial value of eponymous
        attribute.
      **kwargs: Other keyword arguments passed to superclass (`Layer`).
    """
    super().__init__(**kwargs)
    self.input_spec = tf.keras.layers.InputSpec(min_ndim=2)
    self.inverse = inverse
    self.rectify = rectify
    self.data_format = data_format
    self.beta_parameter = beta_parameter
    self.gamma_parameter = gamma_parameter
    self.beta_initializer = beta_initializer
    self.gamma_initializer = gamma_initializer

  def _check_not_built(self):
    if self.built:
      raise RuntimeError(
          "Can't modify layer attributes after it has been built.")

  @property
  def inverse(self) -> bool:
    return self._inverse

  @inverse.setter
  def inverse(self, value):
    self._check_not_built()
    self._inverse = bool(value)

  @property
  def rectify(self) -> bool:
    return self._rectify

  @rectify.setter
  def rectify(self, value):
    self._check_not_built()
    self._rectify = bool(value)

  @property
  def data_format(self) -> str:
    return self._data_format

  @data_format.setter
  def data_format(self, value):
    self._check_not_built()
    value = str(value)
    if value not in ("channels_first", "channels_last"):
      raise ValueError(f"Unknown data format: '{value}'.")
    self._data_format = value

  @property
  def beta_parameter(self) -> Union[None, tf.Tensor, Callable[[], tf.Tensor]]:
    return self._beta_parameter

  @beta_parameter.setter
  def beta_parameter(self, value):
    self._check_not_built()
    # This is necessary to make Keras deserialization via __init__ work.
    if isinstance(value, dict):
      value = tf.keras.utils.deserialize_keras_object(value)
    if value is not None and not callable(value):
      value = tf.convert_to_tensor(value, dtype=self.dtype)
    self._beta_parameter = value

  @property
  def gamma_parameter(self) -> Union[None, tf.Tensor, Callable[[], tf.Tensor]]:
    return self._gamma_parameter

  @gamma_parameter.setter
  def gamma_parameter(self, value):
    self._check_not_built()
    # This is necessary to make Keras deserialization via __init__ work.
    if isinstance(value, dict):
      value = tf.keras.utils.deserialize_keras_object(value)
    if value is not None and not callable(value):
      value = tf.convert_to_tensor(value, dtype=self.dtype)
    self._gamma_parameter = value

  @property
  def beta_initializer(self) -> Callable[..., tf.Tensor]:
    return self._beta_initializer

  @beta_initializer.setter
  def beta_initializer(self, value):
    self._check_not_built()
    self._beta_initializer = tf.keras.initializers.get(value)

  @property
  def gamma_initializer(self) -> Callable[..., tf.Tensor]:
    return self._gamma_initializer

  @gamma_initializer.setter
  def gamma_initializer(self, value):
    self._check_not_built()
    self._gamma_initializer = tf.keras.initializers.get(value)

  @property
  def beta(self) -> tf.Tensor:
    if self.beta_parameter is None:
      raise RuntimeError("beta is not initialized yet. Call build().")
    if callable(self.beta_parameter):
      return tf.convert_to_tensor(self.beta_parameter(), dtype=self.dtype)
    return self.beta_parameter

  @property
  def gamma(self) -> tf.Tensor:
    if self.gamma_parameter is None:
      raise RuntimeError("gamma is not initialized yet. Call build().")
    if callable(self.gamma_parameter):
      return tf.convert_to_tensor(self.gamma_parameter(), dtype=self.dtype)
    return self.gamma_parameter

  @property
  def _channel_axis(self):
    return {"channels_first": 1, "channels_last": -1}[self.data_format]

  def build(self, input_shape):
    channel_axis = self._channel_axis
    input_shape = tf.TensorShape(input_shape)
    num_channels = input_shape[channel_axis]
    if num_channels is None:
      raise ValueError("The channel dimension of the inputs to `GDN` "
                       "must be defined.")
    self.input_spec = tf.keras.layers.InputSpec(
        min_ndim=2, axes={channel_axis: num_channels})

    if self.beta_parameter is None:
      initial_value = self.beta_initializer(
          shape=[num_channels], dtype=self.dtype)
      self.beta_parameter = parameters.GDNParameter(
          initial_value, name="beta", minimum=1e-6)

    if self.gamma_parameter is None:
      initial_value = self.gamma_initializer(
          shape=[num_channels, num_channels], dtype=self.dtype)
      self.gamma_parameter = parameters.GDNParameter(
          initial_value, name="gamma", minimum=0)

    super().build(input_shape)

  def call(self, inputs) -> tf.Tensor:
    inputs = tf.convert_to_tensor(inputs, dtype=self.dtype)
    rank = inputs.shape.rank
    if rank is None:
      raise RuntimeError("Input tensor rank must be defined.")

    if self.rectify:
      inputs = tf.nn.relu(inputs)

    # Compute normalization pool.
    if rank == 2:
      norm_pool = tf.linalg.matmul(tf.math.square(inputs), self.gamma)
      norm_pool = tf.nn.bias_add(norm_pool, self.beta)
    elif self.data_format == "channels_last" and rank <= 5:
      shape = self.gamma.shape
      gamma = tf.reshape(self.gamma, (rank - 2) * [1] + shape)
      norm_pool = tf.nn.convolution(
          tf.math.square(inputs), gamma, padding="VALID")
      norm_pool = tf.nn.bias_add(norm_pool, self.beta)
    else:  # generic implementation
      # This puts channels in the last dimension regardless of input.
      norm_pool = tf.linalg.tensordot(
          tf.math.square(inputs), self.gamma, [[self._channel_axis], [0]])
      norm_pool += self.beta
      if self.data_format == "channels_first":
        # Return to channels_first format if necessary.
        axes = list(range(rank - 1))
        axes.insert(1, rank - 1)
        norm_pool = tf.transpose(norm_pool, axes)

    if self.inverse:
      norm_pool = tf.math.sqrt(norm_pool)
    else:
      norm_pool = tf.math.rsqrt(norm_pool)
    return inputs * norm_pool

  def compute_output_shape(self, input_shape) -> tf.TensorShape:
    return tf.TensorShape(input_shape)

  def get_config(self) -> Dict[str, Any]:
    config = super().get_config()
    config.update(
        inverse=self.inverse,
        rectify=self.rectify,
        data_format=self.data_format,
        beta_parameter=tf.keras.utils.serialize_keras_object(
            self.beta_parameter),
        gamma_parameter=tf.keras.utils.serialize_keras_object(
            self.gamma_parameter),
        beta_initializer=tf.keras.initializers.serialize(
            self.beta_initializer),
        gamma_initializer=tf.keras.initializers.serialize(
            self.gamma_initializer),
    )
    return config
