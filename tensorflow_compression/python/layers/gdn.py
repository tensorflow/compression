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


ParameterType = Union[None, tf.Tensor, Callable[[], tf.Tensor]]


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

  Implements an activation function that is a multivariate generalization of a
  particular sigmoid-type function:

  ```
  y[i] = x[i] / (beta[i] + sum_j(gamma[j, i] * |x[j]|^alpha))^epsilon
  ```

  where `i` and `j` run over channels. This implementation never sums across
  spatial dimensions. It is similar to local response normalization, but much
  more flexible, as `alpha`, `beta`, `gamma`, and `epsilon` are trainable
  parameters.

  Attributes:
    inverse: Boolean. If `False`, compute GDN response. If `True`, compute IGDN
      response (one step of fixed point iteration to invert GDN; the division is
      replaced by multiplication).
    rectify: Boolean. If `True`, apply a `relu` nonlinearity to the inputs
      before calculating GDN response.
    data_format: String. Format of input tensor. Currently supports
      `'channels_first'` and `'channels_last'`.
    alpha_parameter: Scalar, callable, or `None`. A number or scalar `tf.Tensor`
      means that the value of alpha is fixed. A callable can be used to
      determine the value of alpha as a function of some other variable or
      tensor. This can be a `Parameter` object. `None` means that when the layer
      is built, a `GDNParameter` object is created to train alpha (with a
      minimum value of 1). The default is a fixed value of 1. Note that certain
      choices here such as `tf.Tensor`s or lambda functions may prevent
      JSON-style serialization (`Parameter` objects and Python constants work).
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
    epsilon_parameter: Scalar, callable, or `None`. A number or scalar
      `tf.Tensor` means that the value of epsilon is fixed. A callable can be
      used to determine the value of epsilon as a function of some other
      variable or tensor. This can be a `Parameter` object. `None` means that
      when the layer is built, a `GDNParameter` object is created to train
      epsilon (with a minimum value of 1e-6). The default is a fixed value of 1.
      Note that certain choices here such as `tf.Tensor`s or lambda functions
      may prevent JSON-style serialization (`Parameter` objects and Python
      constants work).
    alpha_initializer: `Initializer` object for alpha parameter. Only used if
      alpha is trained. Defaults to 1.
    beta_initializer: `Initializer` object for beta parameter. Only used if beta
      is created on building the layer. Defaults to 1.
    gamma_initializer: `Initializer` object for gamma parameter. Only used if
      gamma is created on building the layer. Defaults to identity matrix
      multiplied by 0.1. A good default value for the diagonal is somewhere
      between 0 and 0.5. If set to 0 and beta initialized as 1, the layer is
      effectively initialized to the identity operation.
    epsilon_initializer: `Initializer` object for epsilon parameter. Only used
      if epsilon is trained. Defaults to 1.
    alpha: `tf.Tensor`. Read-only property always returning the current value of
      alpha.
    beta: `tf.Tensor`. Read-only property always returning the current value of
      beta.
    gamma: `tf.Tensor`. Read-only property always returning the current value of
      gamma.
    epsilon: `tf.Tensor`. Read-only property always returning the current value
      of epsilon.
  """

  def __init__(self,
               inverse=False,
               rectify=False,
               data_format="channels_last",
               alpha_parameter=1,
               beta_parameter=None,
               gamma_parameter=None,
               epsilon_parameter=1,
               alpha_initializer="ones",
               beta_initializer="ones",
               gamma_initializer=tf.keras.initializers.Identity(.1),
               epsilon_initializer="ones",
               **kwargs):
    """Initializer.

    Args:
      inverse: Boolean. Initial value of eponymous attribute.
      rectify: Boolean. Initial value of eponymous attribute.
      data_format: String. Initial value of eponymous attribute.
      alpha_parameter: Scalar, callable, or `None`. Initial value of eponymous
        attribute.
      beta_parameter: Tensor, callable, or `None`. Initial value of eponymous
        attribute.
      gamma_parameter: Tensor, callable, or `None`. Initial value of eponymous
        attribute.
      epsilon_parameter: Scalar, callable, or `None`. Initial value of
        eponymous attribute.
      alpha_initializer: `Initializer` object. Initial value of eponymous
        attribute.
      beta_initializer: `Initializer` object. Initial value of eponymous
        attribute.
      gamma_initializer: `Initializer` object. Initial value of eponymous
        attribute.
      epsilon_initializer: `Initializer` object. Initial value of eponymous
        attribute.
      **kwargs: Other keyword arguments passed to superclass (`Layer`).
    """
    super().__init__(**kwargs)
    self.inverse = inverse
    self.rectify = rectify
    self.data_format = data_format
    self.alpha_parameter = alpha_parameter
    self.beta_parameter = beta_parameter
    self.gamma_parameter = gamma_parameter
    self.epsilon_parameter = epsilon_parameter
    self.alpha_initializer = alpha_initializer
    self.beta_initializer = beta_initializer
    self.gamma_initializer = gamma_initializer
    self.epsilon_initializer = epsilon_initializer

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
  def alpha_parameter(self) -> ParameterType:
    return self._alpha_parameter

  @alpha_parameter.setter
  def alpha_parameter(self, value):
    self._check_not_built()
    # This is necessary to make Keras deserialization via __init__ work.
    if isinstance(value, dict):
      value = tf.keras.utils.deserialize_keras_object(value)
    if value is not None and not callable(value):
      value = tf.convert_to_tensor(value, dtype=self.dtype)
    self._alpha_parameter = value

  @property
  def beta_parameter(self) -> ParameterType:
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
  def gamma_parameter(self) -> ParameterType:
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
  def epsilon_parameter(self) -> ParameterType:
    return self._epsilon_parameter

  @epsilon_parameter.setter
  def epsilon_parameter(self, value):
    self._check_not_built()
    # This is necessary to make Keras deserialization via __init__ work.
    if isinstance(value, dict):
      value = tf.keras.utils.deserialize_keras_object(value)
    if value is not None and not callable(value):
      value = tf.convert_to_tensor(value, dtype=self.dtype)
    self._epsilon_parameter = value

  @property
  def alpha_initializer(self) -> Callable[..., tf.Tensor]:
    return self._alpha_initializer

  @alpha_initializer.setter
  def alpha_initializer(self, value):
    self._check_not_built()
    self._alpha_initializer = tf.keras.initializers.get(value)

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
  def epsilon_initializer(self) -> Callable[..., tf.Tensor]:
    return self._epsilon_initializer

  @epsilon_initializer.setter
  def epsilon_initializer(self, value):
    self._check_not_built()
    self._epsilon_initializer = tf.keras.initializers.get(value)

  @property
  def alpha(self) -> tf.Tensor:
    if self.alpha_parameter is None:
      raise RuntimeError("alpha is not initialized yet. Call build().")
    if callable(self.alpha_parameter):
      return tf.convert_to_tensor(self.alpha_parameter(), dtype=self.dtype)
    return self.alpha_parameter

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
  def epsilon(self) -> tf.Tensor:
    if self.epsilon_parameter is None:
      raise RuntimeError("epsilon is not initialized yet. Call build().")
    if callable(self.epsilon_parameter):
      return tf.convert_to_tensor(self.epsilon_parameter(), dtype=self.dtype)
    return self.epsilon_parameter

  @property
  def _channel_axis(self):
    return {"channels_first": 1, "channels_last": -1}[self.data_format]

  def build(self, input_shape):
    input_shape = tf.TensorShape(input_shape)
    if input_shape.rank is None or input_shape.rank < 2:
      raise ValueError(f"Input tensor must have at least rank 2, received "
                       f"shape {input_shape}.")
    num_channels = input_shape[self._channel_axis]
    if num_channels is None:
      raise ValueError("The channel dimension of the inputs must be defined.")

    if self.alpha_parameter is None:
      initial_value = self.alpha_initializer(
          shape=[], dtype=self.dtype)
      self.alpha_parameter = parameters.GDNParameter(
          initial_value, name="alpha", minimum=1)

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

    if self.epsilon_parameter is None:
      initial_value = self.epsilon_initializer(
          shape=[], dtype=self.dtype)
      self.epsilon_parameter = parameters.GDNParameter(
          initial_value, name="epsilon", minimum=1e-6)

    super().build(input_shape)

  def call(self, inputs) -> tf.Tensor:
    inputs = tf.convert_to_tensor(inputs, dtype=self.dtype)
    rank = inputs.shape.rank
    if rank is None or rank < 2:
      raise ValueError(f"Input tensor must have at least rank 2, received "
                       f"shape {inputs.shape}.")

    if self.rectify:
      inputs = tf.nn.relu(inputs)

    # Optimize for fixed alphas.
    if not callable(self.alpha_parameter) and self.alpha == 1 and self.rectify:
      norm_pool = inputs
    elif not callable(self.alpha_parameter) and self.alpha == 1:
      norm_pool = abs(inputs)
    elif not callable(self.alpha_parameter) and self.alpha == 2:
      norm_pool = tf.math.square(inputs)
    else:
      norm_pool = inputs ** self.alpha

    # Compute normalization pool.
    if rank == 2:
      norm_pool = tf.linalg.matmul(norm_pool, self.gamma)
      norm_pool = tf.nn.bias_add(norm_pool, self.beta)
    elif self.data_format == "channels_last" and rank <= 5:
      shape = self.gamma.shape
      gamma = tf.reshape(self.gamma, (rank - 2) * [1] + shape)
      norm_pool = tf.nn.convolution(norm_pool, gamma, padding="VALID")
      norm_pool = tf.nn.bias_add(norm_pool, self.beta)
    else:  # generic implementation
      # This puts channels in the last dimension regardless of input.
      norm_pool = tf.linalg.tensordot(
          norm_pool, self.gamma, [[self._channel_axis], [0]])
      norm_pool += self.beta
      if self.data_format == "channels_first":
        # Return to channels_first format if necessary.
        axes = list(range(rank - 1))
        axes.insert(1, rank - 1)
        norm_pool = tf.transpose(norm_pool, axes)

    # Optimize for fixed epsilons.
    if not callable(self.epsilon_parameter) and self.epsilon == 1:
      pass
    elif not callable(self.epsilon_parameter) and self.epsilon == .5:
      norm_pool = tf.math.sqrt(norm_pool)
    else:
      norm_pool = norm_pool ** self.epsilon

    if self.inverse:
      return inputs * norm_pool
    else:
      return inputs / norm_pool

  def compute_output_shape(self, input_shape) -> tf.TensorShape:
    return tf.TensorShape(input_shape)

  def get_config(self) -> Dict[str, Any]:
    config = super().get_config()

    # Since alpha and epsilon are scalar, allow fixed values to be serialized.
    def try_serialize(parameter, name):
      if parameter is None:
        return None
      try:
        return tf.keras.utils.serialize_keras_object(parameter)
      except (ValueError, TypeError):  # Should throw TypeError, but doesn't...
        try:
          return float(parameter)
        except TypeError:
          raise TypeError(
              f"Can't serialize {name} of type {type(parameter)}.")

    alpha_parameter = try_serialize(
        self.alpha_parameter, "alpha_parameter")
    epsilon_parameter = try_serialize(
        self.epsilon_parameter, "epsilon_parameter")

    config.update(
        inverse=self.inverse,
        rectify=self.rectify,
        data_format=self.data_format,
        alpha_parameter=alpha_parameter,
        beta_parameter=tf.keras.utils.serialize_keras_object(
            self.beta_parameter),
        gamma_parameter=tf.keras.utils.serialize_keras_object(
            self.gamma_parameter),
        epsilon_parameter=epsilon_parameter,
        alpha_initializer=tf.keras.initializers.serialize(
            self.alpha_initializer),
        beta_initializer=tf.keras.initializers.serialize(
            self.beta_initializer),
        gamma_initializer=tf.keras.initializers.serialize(
            self.gamma_initializer),
        epsilon_initializer=tf.keras.initializers.serialize(
            self.epsilon_initializer),
    )
    return config
