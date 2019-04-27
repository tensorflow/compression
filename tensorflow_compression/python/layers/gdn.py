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
"""GDN layer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow_compression.python.layers import parameterizers


_default_beta_param = parameterizers.NonnegativeParameterizer(
    minimum=1e-6)
_default_gamma_param = parameterizers.NonnegativeParameterizer()


__all__ = [
    "GDN",
]


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

  Arguments:
    inverse: Boolean. If `False` (default), compute GDN response. If `True`,
      compute IGDN response (one step of fixed point iteration to invert GDN;
      the division is replaced by multiplication).
    rectify: Boolean. If `True`, apply a `relu` nonlinearity to the inputs
      before calculating GDN response.
    gamma_init: Float. The gamma matrix will be initialized as the identity
      matrix multiplied with this value. If set to zero, the layer is
      effectively initialized to the identity operation, since beta is
      initialized as one. A good default setting is somewhere between 0 and 0.5.
    data_format: Format of input tensor. Currently supports `'channels_first'`
      and `'channels_last'`.
    beta_parameterizer: `Parameterizer` object for beta parameter. Defaults
      to `NonnegativeParameterizer` with a minimum value of 1e-6.
    gamma_parameterizer: `Parameterizer` object for gamma parameter.
      Defaults to `NonnegativeParameterizer` with a minimum value of 0.
    activity_regularizer: Regularizer function for the output.
    trainable: Boolean. Whether the layer should be trained.
    name: String. The name of the layer.
    dtype: `DType` of the layer's inputs (default of `None` means use the type
      of the first input).

  Read-only properties:
    inverse: Boolean, whether GDN is computed (`True`) or IGDN (`False`).
    rectify: Boolean, whether to apply `relu` before normalization or not.
    gamma_init: See above.
    data_format: See above.
    activity_regularizer: See above.
    name: See above.
    dtype: See above.
    beta: The beta parameter as defined above (1D `Tensor`).
    gamma: The gamma parameter as defined above (2D `Tensor`).
    trainable_variables: List of trainable variables.
    non_trainable_variables: List of non-trainable variables.
    variables: List of all variables of this layer, trainable and non-trainable.
    updates: List of update ops of this layer.
    losses: List of losses added by this layer.

  Mutable properties:
    beta_parameterizer: See above.
    gamma_parameterizer: See above.
    trainable: Boolean. Whether the layer should be trained.
    input_spec: Optional `InputSpec` object specifying the constraints on inputs
      that can be accepted by the layer.
  """

  def __init__(self,
               inverse=False,
               rectify=False,
               gamma_init=.1,
               data_format="channels_last",
               beta_parameterizer=_default_beta_param,
               gamma_parameterizer=_default_gamma_param,
               **kwargs):
    super(GDN, self).__init__(**kwargs)
    self._inverse = bool(inverse)
    self._rectify = bool(rectify)
    self._gamma_init = float(gamma_init)
    self._data_format = str(data_format)
    self._beta_parameterizer = beta_parameterizer
    self._gamma_parameterizer = gamma_parameterizer

    if self.data_format not in ("channels_first", "channels_last"):
      raise ValueError("Unknown data format: '{}'.".format(self.data_format))

    self.input_spec = tf.keras.layers.InputSpec(min_ndim=2)

  @property
  def inverse(self):
    return self._inverse

  @property
  def rectify(self):
    return self._rectify

  @property
  def gamma_init(self):
    return self._gamma_init

  @property
  def data_format(self):
    return self._data_format

  @property
  def beta_parameterizer(self):
    return self._beta_parameterizer

  @beta_parameterizer.setter
  def beta_parameterizer(self, val):
    if self.built:
      raise RuntimeError(
          "Can't set `beta_parameterizer` once layer has been built.")
    self._beta_parameterizer = val

  @property
  def gamma_parameterizer(self):
    return self._gamma_parameterizer

  @gamma_parameterizer.setter
  def gamma_parameterizer(self, val):
    if self.built:
      raise RuntimeError(
          "Can't set `gamma_parameterizer` once layer has been built.")
    self._gamma_parameterizer = val

  def _channel_axis(self):
    return {"channels_first": 1, "channels_last": -1}[self.data_format]

  def build(self, input_shape):
    channel_axis = self._channel_axis()
    input_shape = tf.TensorShape(input_shape)
    num_channels = input_shape[channel_axis].value
    if num_channels is None:
      raise ValueError("The channel dimension of the inputs to `GDN` "
                       "must be defined.")
    self._input_rank = input_shape.ndims
    self.input_spec = tf.keras.layers.InputSpec(
        ndim=input_shape.ndims, axes={channel_axis: num_channels})

    # Sorry, lint, but these objects really are callable ...
    # pylint:disable=not-callable
    self.beta = self.beta_parameterizer(
        name="beta", shape=[num_channels], dtype=self.dtype,
        getter=self.add_variable, initializer=tf.initializers.ones())

    self.gamma = self.gamma_parameterizer(
        name="gamma", shape=[num_channels, num_channels], dtype=self.dtype,
        getter=self.add_variable,
        initializer=tf.initializers.identity(gain=self._gamma_init))
    # pylint:enable=not-callable

    self.built = True

  def call(self, inputs):
    inputs = tf.convert_to_tensor(inputs, dtype=self.dtype)
    ndim = self._input_rank

    if self.rectify:
      inputs = tf.nn.relu(inputs)

    # Compute normalization pool.
    if ndim == 2:
      norm_pool = tf.linalg.matmul(tf.math.square(inputs), self.gamma)
      norm_pool = tf.nn.bias_add(norm_pool, self.beta)
    elif self.data_format == "channels_last" and ndim <= 5:
      shape = self.gamma.shape.as_list()
      gamma = tf.reshape(self.gamma, (ndim - 2) * [1] + shape)
      norm_pool = tf.nn.convolution(tf.math.square(inputs), gamma, "VALID")
      norm_pool = tf.nn.bias_add(norm_pool, self.beta)
    else:  # generic implementation
      # This puts channels in the last dimension regardless of input.
      norm_pool = tf.linalg.tensordot(
          tf.math.square(inputs), self.gamma, [[self._channel_axis()], [0]])
      norm_pool += self.beta
      if self.data_format == "channels_first":
        # Return to channels_first format if necessary.
        axes = list(range(ndim - 1))
        axes.insert(1, ndim - 1)
        norm_pool = tf.transpose(norm_pool, axes)

    if self.inverse:
      norm_pool = tf.math.sqrt(norm_pool)
    else:
      norm_pool = tf.math.rsqrt(norm_pool)
    outputs = inputs * norm_pool

    if not tf.executing_eagerly():
      outputs.set_shape(self.compute_output_shape(inputs.shape))
    return outputs

  def compute_output_shape(self, input_shape):
    return tf.TensorShape(input_shape)
