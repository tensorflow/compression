# Copyright 2022 TensorFlow Compression contributors. All Rights Reserved.
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
"""Sphere process."""

import tensorflow as tf
import tensorflow_probability as tfp


class Sphere(tfp.distributions.Distribution):
  """Uniform distribution on the unit hypersphere."""

  def __init__(self,
               order=2,
               width=0.,
               dtype=tf.float32,
               validate_args=False,
               allow_nan_stats=True,
               name="sphere"):
    """Initializer.

    Arguments:
      order: Integer >= 1. The dimensionality of the sphere.
      width: Float in [0,1]. Allows for realizations to be approximately
        uniformly distributed in a band between radius `1 - width` and
        `1 + width` (for `width` << 1).
      dtype: Data type of the returned realization. Defaults to `tf.float32`.
      validate_args: required by `Distribution` class but unused.
      allow_nan_stats: required by `Distribution` class but unused.
      name: String. Name of the created object.
    """
    parameters = dict(locals())
    self._order = int(order)
    self._width = float(width)
    super().__init__(
        dtype=dtype,
        reparameterization_type=tfp.distributions.NOT_REPARAMETERIZED,
        validate_args=validate_args,
        allow_nan_stats=allow_nan_stats,
        parameters=parameters,
        name=name,
    )

  @property
  def order(self):
    return self._order

  @property
  def width(self):
    return self._width

  def _batch_shape_tensor(self):
    return tf.constant([], dtype=tf.int32)

  def _batch_shape(self):
    return tf.TensorShape([])

  def _event_shape_tensor(self):
    return tf.constant([self.order], dtype=tf.int32)

  def _event_shape(self):
    return tf.TensorShape([self.order])

  def _sample_n(self, n, seed=None):
    samples = tf.random.stateless_normal(
        (n, self.order), seed=seed, dtype=self.dtype)
    radius = tf.math.sqrt(tf.reduce_sum(tf.square(samples), -1, keepdims=True))
    if self.width:
      radius *= tf.random.stateless_uniform(
          (n, 1), minval=1. - self.width / 2., maxval=1. + self.width / 2.,
          seed=seed, dtype=self.dtype)
    return samples / radius
