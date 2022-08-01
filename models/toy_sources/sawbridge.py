# Copyright 2020 TensorFlow Compression contributors. All Rights Reserved.
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
"""Sawbridge process."""

import tensorflow as tf
import tensorflow_probability as tfp


class Sawbridge(tfp.distributions.Distribution):
  """The "sawbridge": B(t) = t - 1(t > Z), where Z is uniform over [0,1].

  The stationary sawbridge is B(t + V mod 1), where V is uniform over [0,1] and
  Z, V are independent.
  """

  def __init__(self, index_points, phase=None, drop=None, stationary=True,
               order=1, dtype=tf.float32, validate_args=False,
               allow_nan_stats=True, name="sawbridge"):
    """Initializer.

    Args:
      index_points: 1-D `Tensor` representing the locations at which to evaluate
        the process. The intent is that all locations are in [0,1], but the
        process has a natural extrapolation outside this range so no error is
        thrown.
      phase: Float in [0,1] or `None` (default). Specifies realization of V.
      drop: Float in [0,1] or `None` (default). Specifies realization of Z.
      stationary: Boolean. Whether or not to "scramble" phase.
      order: Integer >= 1. The resulting process is a linear combination of
        `order` sawbridges.
      dtype: Data type of the returned realization. Defaults to `tf.float32`.
      validate_args: required by `Distribution` class but unused.
      allow_nan_stats: required by `Distribution` class but unused.
      name: String. Name of the created object.
    """
    parameters = dict(locals())
    with tf.name_scope(name) as name:
      self._index_points = tf.convert_to_tensor(
          index_points, dtype_hint=dtype, name="index_points")
      self._stationary = bool(stationary)
      self._order = int(order)
      self._phase = phase
      self._drop = drop
    super().__init__(
        dtype=dtype,
        reparameterization_type=tfp.distributions.NOT_REPARAMETERIZED,
        validate_args=validate_args,
        allow_nan_stats=allow_nan_stats,
        parameters=parameters,
        name=name,
    )

  @property
  def index_points(self):
    return self._index_points

  @property
  def stationary(self):
    return self._stationary

  @property
  def order(self):
    return self._order

  @property
  def phase(self):
    return self._phase

  @property
  def drop(self):
    return self._drop

  def _batch_shape_tensor(self):
    return tf.constant([], dtype=tf.int32)

  def _batch_shape(self):
    return tf.TensorShape([])

  def _event_shape_tensor(self):
    return tf.shape(self.index_points)

  def _event_shape(self):
    return self.index_points.shape

  def _sample_n(self, n, seed=None):
    if self.drop is None:
      uniform = tf.random.stateless_uniform(
          (self.order, n, 1), seed=seed, dtype=self.dtype)
    else:
      uniform = tf.fill(
          (self.order, n, 1), tf.constant(self.drop, dtype=self.dtype))
    ind = self.index_points
    if self.stationary:
      if self.phase is None:
        phase = tf.random.stateless_uniform((n, 1), seed=seed, dtype=self.dtype)
      else:
        phase = tf.constant(self.phase, dtype=self.dtype)
      ind += phase
      ind %= 1.
    less = tf.less(uniform, ind)  # shape: (order, n, time)
    # Note:
    # ind[n] == 1 -> sample[n] == 0 always
    # ind[n] == 0 -> sample[n] == 0 always
    sample = ind - tf.reduce_sum(tf.cast(less, self.dtype), 0)
    # Divide by sqrt(order).
    sample *= self.order ** -.5
    return sample
