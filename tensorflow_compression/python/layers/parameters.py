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
"""Parameters for layer classes."""

import abc
from typing import Any, Dict
import tensorflow as tf
from tensorflow_compression.python.ops import math_ops


__all__ = [
    "Parameter",
    "RDFTParameter",
    "GDNParameter",
]


class Parameter(tf.Module, metaclass=abc.ABCMeta):
  """Reparameterized `Layer` variable.

  This object represents a parameter of a `tf.keras.layer.Layer` object which
  isn't directly stored in a `tf.Variable`, but can be represented as a function
  (of any number of `tf.Variable` attributes).
  """

  @abc.abstractmethod
  def __call__(self):
    """Computes and returns the parameter value as a `tf.Tensor`."""

  @abc.abstractmethod
  def get_config(self):
    """Returns the configuration of the `Parameter`."""
    return dict(name=self.name)

  def get_weights(self):
    return tf.keras.backend.batch_get_value(self.variables)

  def set_weights(self, weights):
    if len(weights) != len(self.variables):
      raise ValueError(
          f"set_weights() expects a list of {len(self.variables)} arrays, "
          f"received {len(weights)}.")
    tf.keras.backend.batch_set_value(zip(self.variables, weights))


def _parameter_conversion_func(value, dtype=None, name=None, as_ref=False):
  if as_ref:
    raise ValueError("as_ref=True is not supported.")
  return tf.convert_to_tensor(value(), dtype=dtype, name=name)


tf.register_tensor_conversion_function(
    Parameter, _parameter_conversion_func,
)


@tf.keras.utils.register_keras_serializable(package="tensorflow_compression")
class RDFTParameter(Parameter):
  """RDFT reparameterization of a convolution kernel.

  This uses the real-input discrete Fourier transform (RDFT) of a kernel as
  its parameterization. The inverse RDFT is applied to the variable to produce
  the kernel.

  (see https://en.wikipedia.org/wiki/Discrete_Fourier_transform)

  Attributes:
    shape: `tf.TensorShape`. The shape of the convolution kernel.
    real: `tf.Variable`. The real part of the RDFT of the kernel.
    imag: `tf.Variable`. The imaginary part of the RDFT of the kernel.
  """

  def __init__(self, initial_value, name=None, shape=None, dtype=None):
    """Initializer.

    Args:
      initial_value: `tf.Tensor` or `None`. The initial value of the kernel. If
        not provided, its `shape` must be given, and the initial value of the
        parameter will be undefined.
      name: String. The name of the kernel.
      shape: `tf.TensorShape` or compatible. Ignored unless `initial_value is
        None`.
      dtype: `tf.dtypes.DType` or compatible. DType of this parameter. If not
        given, inferred from `initial_value`.
    """
    super().__init__(name=name)
    if initial_value is None:
      if shape is None:
        raise ValueError("If initial_value is None, shape must be specified.")
      initial_value = tf.zeros(shape, dtype=dtype)
    else:
      initial_value = tf.convert_to_tensor(initial_value, dtype=dtype)
    self._shape = initial_value.shape
    self._dtype = initial_value.dtype

    if self.shape.rank == 3:
      initial_value = tf.transpose(initial_value, (1, 2, 0))
      initial_value = tf.signal.rfft(initial_value)
    elif self.shape.rank == 4:
      initial_value = tf.transpose(initial_value, (2, 3, 0, 1))
      initial_value = tf.signal.rfft2d(initial_value)
    elif self.shape.rank == 5:
      initial_value = tf.transpose(initial_value, (3, 4, 0, 1, 2))
      initial_value = tf.signal.rfft3d(initial_value)
    else:
      raise ValueError(
          f"Expected kernel tensor of rank 3, 4, or 5; received shape "
          f"{self._shape}.")
    self._norm = tf.constant(
        self.shape[:-2].num_elements() ** .5, initial_value.dtype)
    initial_value /= self._norm
    # We split the variable into real and imaginary parts to avoid issues with
    # complex-valued variables being unsupported when saving models, etc.
    real = tf.math.real(initial_value)
    imag = tf.math.imag(initial_value)
    real_name = imag_name = None
    if name is not None:
      real_name = f"{name}_real"
      imag_name = f"{name}_imag"
    self.real = tf.Variable(real, name=real_name)
    self.imag = tf.Variable(imag, name=imag_name)

  @property
  def dtype(self) -> tf.dtypes.DType:
    return self._dtype

  @property
  def shape(self) -> tf.TensorShape:
    return self._shape

  @tf.Module.with_name_scope
  def __call__(self) -> tf.Tensor:
    """Computes and returns the convolution kernel as a `tf.Tensor`."""
    rdft = tf.dtypes.complex(self.real, self.imag) * self._norm
    if self.shape.rank == 3:
      kernel = tf.signal.irfft(rdft, fft_length=self.shape[:-2])
      return tf.transpose(kernel, (2, 0, 1))
    elif self.shape.rank == 4:
      kernel = tf.signal.irfft2d(rdft, fft_length=self.shape[:-2])
      return tf.transpose(kernel, (2, 3, 0, 1))
    else:
      assert self.shape.rank == 5, self.shape
      kernel = tf.signal.irfft3d(rdft, fft_length=self.shape[:-2])
      return tf.transpose(kernel, (2, 3, 4, 0, 1))

  def get_config(self) -> Dict[str, Any]:
    config = super().get_config()
    config.update(
        initial_value=None,
        shape=tuple(map(int, self.shape)),
        dtype=self.dtype.name,
    )
    return config


@tf.keras.utils.register_keras_serializable(package="tensorflow_compression")
class GDNParameter(Parameter):
  """Nonnegative parameterization as needed for GDN parameters.

  The variable is subjected to an invertible transformation that slows down the
  learning rate for small values.

  Attributes:
    minimum: Float. The `minimum` parameter provided on initialization.
    offset: Float. The `offset` parameter provided on initialization.
    variable: `tf.Variable`. The reparameterized variable.
  """

  def __init__(self, initial_value, name=None, minimum=0., offset=2 ** -18,
               shape=None, dtype=None):
    """Initializer.

    Args:
      initial_value: `tf.Tensor` or `None`. The initial value of the kernel. If
        not provided, its `shape` must be given, and the initial value of the
        parameter will be undefined.
      name: String. The name of the parameter.
      minimum: Float. Lower bound for the parameter (defaults to zero).
      offset: Float. Offset added to the reparameterization. The
        parameterization of beta/gamma as their square roots lets the training
        slow down when values are close to zero, which is desirable as small
        values in the denominator can lead to a situation where gradient noise
        on beta/gamma leads to extreme amounts of noise in the GDN activations.
        However, without the offset, we would get zero gradients if any elements
        of beta or gamma were exactly zero, and thus the training could get
        stuck. To prevent this, we add this small constant. The default value
        was empirically determined as a good starting point. Making it bigger
        potentially leads to more gradient noise on the activations, making it
        too small may lead to numerical precision issues.
      shape: `tf.TensorShape` or compatible. Ignored unless `initial_value is
        None`.
      dtype: `tf.dtypes.DType` or compatible. DType of this parameter. If not
        given, inferred from `initial_value`.
    """
    super().__init__(name=name)
    self._minimum = float(minimum)
    self._offset = float(offset)
    if initial_value is None:
      if shape is None:
        raise ValueError("If initial_value is None, shape must be specified.")
      initial_value = tf.zeros(shape, dtype=dtype)
    else:
      initial_value = tf.convert_to_tensor(initial_value, dtype=dtype)
    self._pedestal = tf.constant(self.offset ** 2, dtype=initial_value.dtype)
    self._bound = tf.constant(
        (self.minimum + self.offset ** 2) ** .5, dtype=initial_value.dtype)
    initial_value = tf.math.sqrt(
        tf.math.maximum(initial_value + self._pedestal, self._pedestal))
    if name is not None:
      name = f"reparam_{name}"
    self.variable = tf.Variable(initial_value, name=name)

  @tf.Module.with_name_scope
  def __call__(self) -> tf.Tensor:
    """Computes and returns the non-negative value as a `tf.Tensor`."""
    reparam_value = math_ops.lower_bound(self.variable, self._bound)
    return tf.math.square(reparam_value) - self._pedestal

  @property
  def minimum(self) -> float:
    return self._minimum

  @property
  def offset(self) -> float:
    return self._offset

  def get_config(self) -> Dict[str, Any]:
    config = super().get_config()
    config.update(
        initial_value=None,
        minimum=self.minimum,
        offset=self.offset,
        shape=tuple(map(int, self.variable.shape)),
        dtype=self.variable.dtype.name,
    )
    return config
