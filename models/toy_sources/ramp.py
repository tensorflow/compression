"""Ramp process."""

import tensorflow as tf
import tensorflow_probability as tfp

class Ramp(tfp.distributions.Distribution):
  """The "ramp": Y(t) = (t+V)mod1 - 0.5, where V is uniform over [0,1]. """
  def __init__(self,index_points,phase=None,dtype=tf.float32,
                 validate_args=False,allow_nan_stats=True,name='ramp'):
    """
    Args:
    index_points: 1-D Tensor representing the locations at which to evaluate
    the process.
    phase: Float in [0,1]. Specifies a realization of V.
    dtype: Data type of the returned realization. Defaults to tf.flaot32.
    validate_args: required by tensorflow Distribution class but not used.
    allow_nan_stats: required by tensorflow Distribution class but not used.
    name: String. Name of the created object.
    """
    parameters = dict(locals())
    with tf.name_scope(name) as name:
      self._index_points = tf.convert_to_tensor(index_points,dtype_hint= dtype,
                                                name='index_points')
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
    if self._phase is None:
      ind += tf.random.uniform ((n,1),seed=seed,dtype=self.dtype)
      ind %= 1
    else:
      phase = tf.repeat(self._phase,repeats=n,axis=0)
      phase = tf.reshape(phase,[n,1])
      ind += phase
      ind %= 1
      return ind - 0.5
