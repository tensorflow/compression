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
"""Ramp process."""

import tensorflow as tf
import tensorflow_probability as tfp


class Ramp(tfp.distributions.Distribution):
  """The "ramp": Y(t) = (t+V) mod 1 - 0.5, where V is uniform over [0, 1]."""

  def __init__(self,
               index_points,
               phase=None,
               dtype=tf.float32,
               validate_args=False,
               allow_nan_stats=True,
               name="ramp"):
    """Initializer.

    Args:
      index_points: 1-D Tensor representing the locations at which to evaluate
        the process.
      phase: Float in [0,1]. Specifies a realization of V.
      dtype: Data type of the returned realization. Defaults to `tf.float32`.
      validate_args: required by `Distribution` class but unused.
      allow_nan_stats: required by `Distribution` class but unused.
      name: String. Name of the created object.
    """
    parameters = dict(locals())
    with tf.name_scope(name) as name:
      self._index_points = tf.convert_to_tensor(
          index_points, dtype_hint=dtype, name="index_points")
      self._phase = phase
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
  def phase(self):
    return self._phase

  def _batch_shape_tensor(self):
    return tf.constant([], dtype=tf.int32)

  def _batch_shape(self):
    return tf.TensorShape([])

  def _event_shape_tensor(self):
    return tf.shape(self.index_points)

  def _event_shape(self):
    return self.index_points.shape

  def _sample_n(self, n, seed=None):
    ind = self.index_points
    if self.phase is None:
      phase = tf.random.stateless_uniform((n, 1), seed=seed, dtype=self.dtype)
    else:
      phase = tf.fill((n, 1), tf.constant(self.phase, dtype=self.dtype))
    return (ind + phase) % 1 - .5
