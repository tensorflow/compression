"""Sinusoid process."""

import tensorflow as tf
import numpy as np
import tensorflow_probability as tfp


class Sinusoid(tfp.distributions.Distribution):
  """The "sinusoid": P(t) = sin(2pi(t+V)), where V is uniform over [0,1]."""

  def __init__(self, index_points, phase=None, dtype=tf.float32,
               validate_args=False, allow_nan_stats=True, name="sinusoid"):
    """Initializer.

    Args:
      index_points: 1-D `Tensor` representing the locations at which to
        evaluate the process. The intent is that all locations are in [0,1],
        but the process has a natural extrapolation outside this range so no
        error is thrown.
      phase: Float in [0,1]. Specifies realization of V.
      dtype: Data type of the returned realization at each timestep. Defaults
        to tf.float32.
      validate_args: required by tensorflow Distribution class but unused.
      allow_nan_stats: required by tensorflow Distribution class but unused.
      name: String. Name of the created object.
    """
    parameters = dict(locals())
    with tf.name_scope(name) as name:
      self._index_points = tf.convert_to_tensor(index_points,dtype_hint= dtype,
                                              name="index_points")
      self._phase = phase
    super().__init__(
      dtype=dtype,
      reparameterization_type = tfp.distributions.NOT_REPARAMETERIZED,
      validate_args = validate_args,
      allow_nan_stats = allow_nan_stats,
      parameters = parameters,
      name=name,)

  @property
  def index_points(self):
    return self._index_points

  @property
  def phase(self):
    return self._phase

  def _batch_shape_tensor(self):
    return tf.constant([],dtype=tf.int32)

  def _batch_shape(self):
    return tf.TensorShape([])

  def _event_shape(self):
    return self.index_points.shape

  def _event_shape_tensor(self):
    return tf.shape(self.index_points)

  def _sample_n(self,n,seed=None):
    ind = self.index_points
    if self.phase is None:
      ind = tf.sin(2*np.pi*(ind+tf.random.uniform((n,1),seed=seed,
                                                  dtype=self.dtype)))
    else:
      phase = tf.repeat(self._phase,repeats=n,axis=0)
      phase = tf.reshape(phase,[n,1])
      ind = tf.sin(2*np.pi(ind+phase))
      return ind
