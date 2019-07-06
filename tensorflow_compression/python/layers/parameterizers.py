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
"""Parameterizers for layer classes."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow_compression.python.ops import math_ops
from tensorflow_compression.python.ops import spectral_ops


__all__ = [
    "Parameterizer",
    "StaticParameterizer",
    "RDFTParameterizer",
    "NonnegativeParameterizer",
]


class Parameterizer(object):
  """Parameterization object (abstract base class).

  `Parameterizer`s are immutable objects designed to facilitate
  reparameterization of model parameters (tensor variables). They are called
  just like `tf.get_variable` with an additional argument `getter` specifying
  the actual function call to generate a variable (in many cases, `getter` would
  be `tf.get_variable`).

  To achieve reparameterization, a `Parameterizer` wraps the provided
  initializer, regularizer, and the returned variable in its own TensorFlow
  code.
  """
  pass


class StaticParameterizer(Parameterizer):
  """A parameterizer that returns a non-variable.

  No variables are created, and `getter` is ignored. If `value` is a `Tensor`,
  the parameter can depend on some other computation. Otherwise, it never
  changes.

  Args:
    value: Either a constant or `Tensor` value, or a callable which returns such
      a thing given a shape and dtype argument (for example, an initializer).
  """

  def __init__(self, value):
    self.value = value

  def __call__(self, getter, name, shape, dtype, initializer, regularizer=None):
    del getter, name, initializer  # unused
    if regularizer is not None:
      raise NotImplementedError("Regularizers are not currently supported for "
                                "static parameterizers.")
    if callable(self.value):
      # Treat value as initializer.
      return self.value(shape, dtype=dtype)
    else:
      return self.value


class RDFTParameterizer(Parameterizer):
  """Object encapsulating RDFT reparameterization.

  This uses the real-input discrete Fourier transform (RDFT) of a kernel as
  its parameterization. The inverse RDFT is applied to the variable to produce
  the parameter.

  (see https://en.wikipedia.org/wiki/Discrete_Fourier_transform)

  Args:
    dc: Boolean. If `False`, the DC component of the kernel RDFTs is not
      represented, forcing the filters to be highpass. Defaults to `True`.
  """

  def __init__(self, dc=True):
    self.dc = bool(dc)

  def __call__(self, getter, name, shape, dtype, initializer, regularizer=None):
    if all(s == 1 for s in shape[:-2]):
      return getter(name=name, shape=shape, dtype=dtype,
                    initializer=initializer, regularizer=regularizer)
    var_shape = shape
    var_dtype = dtype
    size = var_shape[0]
    for s in var_shape[1:-2]:
      size *= s
    if self.dc:
      rdft_shape = (size, var_shape[-2] * var_shape[-1])
    else:
      rdft_shape = (size - 1, var_shape[-2] * var_shape[-1])
    rdft_dtype = var_dtype
    rdft_name = name + "_rdft"

    def rdft_initializer(shape, dtype=None, partition_info=None):
      """Initializer wrapper."""
      assert tuple(shape) == rdft_shape, shape
      assert dtype == rdft_dtype, dtype
      init = initializer(
          var_shape, dtype=var_dtype, partition_info=partition_info)
      init = tf.reshape(init, (-1, rdft_shape[-1]))
      irdft_matrix = spectral_ops.irdft_matrix(var_shape[:-2], dtype=var_dtype)
      if not self.dc:
        irdft_matrix = irdft_matrix[:, 1:]
      init = tf.linalg.matmul(irdft_matrix, init, transpose_a=True)
      return init

    def reparam(rdft):
      irdft_matrix = spectral_ops.irdft_matrix(var_shape[:-2], dtype=var_dtype)
      if not self.dc:
        irdft_matrix = irdft_matrix[:, 1:]
      var = tf.linalg.matmul(irdft_matrix, rdft)
      var = tf.reshape(var, var_shape)
      return var

    reparam_regularizer = None
    if regularizer is not None:
      reparam_regularizer = lambda rdft: regularizer(reparam(rdft))

    rdft = getter(
        name=rdft_name, shape=rdft_shape, dtype=rdft_dtype,
        initializer=rdft_initializer, regularizer=reparam_regularizer)
    return reparam(rdft)


class NonnegativeParameterizer(Parameterizer):
  """Object encapsulating nonnegative parameterization as needed for GDN.

  The variable is subjected to an invertible transformation that slows down the
  learning rate for small values.

  Args:
    minimum: Float. Lower bound for parameters (defaults to zero).
    reparam_offset: Float. Offset added to the reparameterization of beta and
      gamma. The parameterization of beta and gamma as their square roots lets
      the training slow down when their values are close to zero, which is
      desirable as small values in the denominator can lead to a situation where
      gradient noise on beta/gamma leads to extreme amounts of noise in the GDN
      activations. However, without the offset, we would get zero gradients if
      any elements of beta or gamma were exactly zero, and thus the training
      could get stuck. To prevent this, we add this small constant. The default
      value was empirically determined as a good starting point. Making it
      bigger potentially leads to more gradient noise on the activations, making
      it too small may lead to numerical precision issues.
  """

  def __init__(self, minimum=0, reparam_offset=2 ** -18):
    self.minimum = float(minimum)
    self.reparam_offset = float(reparam_offset)

  def __call__(self, getter, name, shape, dtype, initializer, regularizer=None):
    pedestal = tf.constant(self.reparam_offset ** 2, dtype=dtype)
    bound = tf.constant(
        (self.minimum + self.reparam_offset ** 2) ** .5, dtype=dtype)
    reparam_name = "reparam_" + name

    def reparam_initializer(shape, dtype=None, partition_info=None):
      init = initializer(shape, dtype=dtype, partition_info=partition_info)
      init = tf.math.sqrt(tf.math.maximum(init + pedestal, pedestal))
      return init

    def reparam(var):
      var = math_ops.lower_bound(var, bound)
      var = tf.math.square(var) - pedestal
      return var

    reparam_regularizer = None
    if regularizer is not None:
      reparam_regularizer = lambda var: regularizer(reparam(var))

    var = getter(
        name=reparam_name, shape=shape, dtype=dtype,
        initializer=reparam_initializer, regularizer=reparam_regularizer)
    return reparam(var)
