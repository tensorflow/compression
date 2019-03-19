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

# Dependency imports

from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.layers import base
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn

from tensorflow_compression.python.layers import parameterizers


_default_beta_param = parameterizers.NonnegativeParameterizer(
    minimum=1e-6)
_default_gamma_param = parameterizers.NonnegativeParameterizer()


class GDN(base.Layer):
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
  y[i] = x[i] / sqrt(beta[i] + sum_j(gamma[j, i] * x[j]))
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
    gamma_init: The gamma matrix will be initialized as the identity matrix
      multiplied with this value. If set to zero, the layer is effectively
      initialized to the identity operation, since beta is initialized as one.
      A good default setting is somewhere between 0 and 0.5.
    data_format: Format of input tensor. Currently supports `'channels_first'`
      and `'channels_last'`.
    beta_parameterizer: Reparameterization for beta parameter. Defaults to
      `NonnegativeParameterizer` with a minimum value of `1e-6`.
    gamma_parameterizer: Reparameterization for gamma parameter. Defaults to
      `NonnegativeParameterizer` with a minimum value of `0`.
    activity_regularizer: Regularizer function for the output.
    trainable: Boolean, if `True`, also add variables to the graph collection
      `GraphKeys.TRAINABLE_VARIABLES` (see `tf.Variable`).
    name: String, the name of the layer. Layers with the same name will
      share weights, but to avoid mistakes we require `reuse=True` in such
      cases.

  Properties:
    inverse: Boolean, whether GDN is computed (`True`) or IGDN (`False`).
    rectify: Boolean, whether to apply `relu` before normalization or not.
    data_format: Format of input tensor. Currently supports `'channels_first'`
      and `'channels_last'`.
    beta: The beta parameter as defined above (1D `Tensor`).
    gamma: The gamma parameter as defined above (2D `Tensor`).
  """

  def __init__(self,
               inverse=False,
               rectify=False,
               gamma_init=.1,
               data_format="channels_last",
               beta_parameterizer=_default_beta_param,
               gamma_parameterizer=_default_gamma_param,
               activity_regularizer=None,
               trainable=True,
               name=None,
               **kwargs):
    super(GDN, self).__init__(trainable=trainable, name=name,
                              activity_regularizer=activity_regularizer,
                              **kwargs)
    self.inverse = bool(inverse)
    self.rectify = bool(rectify)
    self._gamma_init = float(gamma_init)
    self.data_format = data_format
    self._beta_parameterizer = beta_parameterizer
    self._gamma_parameterizer = gamma_parameterizer
    self._channel_axis()  # trigger ValueError early
    self.input_spec = base.InputSpec(min_ndim=2)

  def _channel_axis(self):
    try:
      return {"channels_first": 1, "channels_last": -1}[self.data_format]
    except KeyError:
      raise ValueError("Unsupported `data_format` for GDN layer: {}.".format(
          self.data_format))

  def build(self, input_shape):
    channel_axis = self._channel_axis()
    input_shape = tensor_shape.TensorShape(input_shape)
    num_channels = input_shape[channel_axis].value
    if num_channels is None:
      raise ValueError("The channel dimension of the inputs to `GDN` "
                       "must be defined.")
    self._input_rank = input_shape.ndims
    self.input_spec = base.InputSpec(ndim=input_shape.ndims,
                                     axes={channel_axis: num_channels})

    self.beta = self._beta_parameterizer(
        name="beta", shape=[num_channels], dtype=self.dtype,
        getter=self.add_variable, initializer=init_ops.Ones())

    self.gamma = self._gamma_parameterizer(
        name="gamma", shape=[num_channels, num_channels], dtype=self.dtype,
        getter=self.add_variable,
        initializer=init_ops.Identity(gain=self._gamma_init))

    self.built = True

  def call(self, inputs):
    inputs = ops.convert_to_tensor(inputs, dtype=self.dtype)
    ndim = self._input_rank

    if self.rectify:
      inputs = nn.relu(inputs)

    # Compute normalization pool.
    if ndim == 2:
      norm_pool = math_ops.matmul(math_ops.square(inputs), self.gamma)
      norm_pool = nn.bias_add(norm_pool, self.beta)
    elif self.data_format == "channels_last" and ndim <= 5:
      shape = self.gamma.shape.as_list()
      gamma = array_ops.reshape(self.gamma, (ndim - 2) * [1] + shape)
      norm_pool = nn.convolution(math_ops.square(inputs), gamma, "VALID")
      norm_pool = nn.bias_add(norm_pool, self.beta)
    else:  # generic implementation
      # This puts channels in the last dimension regardless of input.
      norm_pool = math_ops.tensordot(
          math_ops.square(inputs), self.gamma, [[self._channel_axis()], [0]])
      norm_pool += self.beta
      if self.data_format == "channels_first":
        # Return to channels_first format if necessary.
        axes = list(range(ndim - 1))
        axes.insert(1, ndim - 1)
        norm_pool = array_ops.transpose(norm_pool, axes)

    if self.inverse:
      norm_pool = math_ops.sqrt(norm_pool)
    else:
      norm_pool = math_ops.rsqrt(norm_pool)
    outputs = inputs * norm_pool

    if not context.executing_eagerly():
      outputs.set_shape(self.compute_output_shape(inputs.shape))
    return outputs

  def compute_output_shape(self, input_shape):
    return tensor_shape.TensorShape(input_shape)
