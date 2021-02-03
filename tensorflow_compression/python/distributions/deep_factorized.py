# Copyright 2020 Google LLC. All Rights Reserved.
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
"""Deep fully factorized distribution based on cumulative."""

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_compression.python.distributions import helpers
from tensorflow_compression.python.distributions import uniform_noise


__all__ = [
    "DeepFactorized",
    "NoisyDeepFactorized",
]


def log_expm1(x):
  """Computes log(exp(x)-1) stably.

  For large values of x, exp(x) will return Inf whereas log(exp(x)-1) ~= x.
  Here we use this approximation for x>15, such that the output is non-Inf for
  all positive values x.

  Args:
   x: A tensor.

  Returns:
    log(exp(x)-1)

  """
  # If x<15.0, we can compute it directly. For larger values,
  # we have log(exp(x)-1) ~= log(exp(x)) = x.
  cond = (x < 15.0)
  x_small = tf.minimum(x, 15.0)
  return tf.where(cond, tf.math.log(tf.math.expm1(x_small)), x)


class DeepFactorized(tfp.distributions.Distribution):
  """Fully factorized distribution based on neural network cumulative.

  This is a flexible, nonparametric probability density model, described in
  appendix 6.1 of the paper:

  > "Variational image compression with a scale hyperprior"<br />
  > J. Ballé, D. Minnen, S. Singh, S. J. Hwang, N. Johnston<br />
  > https://openreview.net/forum?id=rkcQFMZRb

  but *without* convolution with a unit-width uniform
  density, as described in appendix 6.2 of the same paper. Please cite the paper
  if you use this code for scientific work.

  This is a scalar distribution (i.e., its `event_shape` is always length 0),
  and the density object always creates its own `tf.Variable`s representing the
  trainable distribution parameters.
  """

  def __init__(self,
               batch_shape=(), num_filters=(3, 3), init_scale=10,
               allow_nan_stats=False, dtype=tf.float32, name="DeepFactorized"):
    """Initializer.

    Args:
      batch_shape: Iterable of integers. The desired batch shape for the
        `Distribution` (rightmost dimensions which are assumed independent, but
        not identically distributed).
      num_filters: Iterable of integers. The number of filters for each of the
        hidden layers. The first and last layer of the network implementing the
        cumulative distribution are not included (they are assumed to be 1).
      init_scale: Float. Scale factor for the density at initialization. It is
        recommended to choose a large enough scale factor such that most values
        initially lie within a region of high likelihood. This improves
        training.
      allow_nan_stats: Boolean. Whether to allow `NaN`s to be returned when
        querying distribution statistics.
      dtype: A floating point `tf.dtypes.DType`. Computations relating to this
        distribution will be performed at this precision.
      name: String. A name for this distribution.
    """
    parameters = dict(locals())
    self._batch_shape_tuple = tuple(int(s) for s in batch_shape)
    self._num_filters = tuple(int(f) for f in num_filters)
    self._init_scale = float(init_scale)
    super().__init__(
        dtype=dtype,
        reparameterization_type=tfp.distributions.NOT_REPARAMETERIZED,
        validate_args=False,
        allow_nan_stats=allow_nan_stats,
        parameters=parameters,
        name=name,
    )
    with self.name_scope:
      self._make_variables()

  @property
  def num_filters(self):
    return self._num_filters

  @property
  def init_scale(self):
    return self._init_scale

  def _make_variables(self):
    """Creates the variables representing the parameters of the distribution."""
    channels = self.batch_shape.num_elements()
    filters = (1,) + self.num_filters + (1,)
    scale = self.init_scale ** (1 / (len(self.num_filters) + 1))
    self._matrices = []
    self._biases = []
    self._factors = []

    for i in range(len(self.num_filters) + 1):

      def matrix_initializer(i=i):
        init = log_expm1(1 / scale / filters[i + 1])
        init = tf.cast(init, dtype=self.dtype)
        init = tf.broadcast_to(init, (channels, filters[i + 1], filters[i]))
        return init

      matrix = tf.Variable(matrix_initializer, name="matrix_{}".format(i))
      self._matrices.append(matrix)

      def bias_initializer(i=i):
        return tf.random.uniform(
            (channels, filters[i + 1], 1), -.5, .5, dtype=self.dtype)

      bias = tf.Variable(bias_initializer, name="bias_{}".format(i))
      self._biases.append(bias)

      if i < len(self.num_filters):

        def factor_initializer(i=i):
          return tf.zeros((channels, filters[i + 1], 1), dtype=self.dtype)

        factor = tf.Variable(factor_initializer, name="factor_{}".format(i))
        self._factors.append(factor)

  def _batch_shape_tensor(self):
    return tf.constant(self._batch_shape_tuple, dtype=int)

  def _batch_shape(self):
    return tf.TensorShape(self._batch_shape_tuple)

  def _event_shape_tensor(self):
    return tf.constant((), dtype=int)

  def _event_shape(self):
    return tf.TensorShape(())

  def _broadcast_inputs(self, inputs):
    shape = tf.broadcast_dynamic_shape(
        tf.shape(inputs), self.batch_shape_tensor())
    return tf.broadcast_to(inputs, shape)

  def _logits_cumulative(self, inputs):
    """Evaluate logits of the cumulative densities.

    Args:
      inputs: The values at which to evaluate the cumulative densities.

    Returns:
      A `tf.Tensor` of the same shape as `inputs`, containing the logits of the
      cumulative densities evaluated at the given inputs.
    """
    # Convert to (channels, 1, batch) format by collapsing dimensions and then
    # commuting channels to front.
    shape = tf.shape(inputs)
    inputs = tf.reshape(inputs, (-1, 1, self.batch_shape.num_elements()))
    inputs = tf.transpose(inputs, (2, 1, 0))
    logits = inputs
    for i in range(len(self.num_filters) + 1):
      matrix = tf.nn.softplus(self._matrices[i])
      logits = tf.linalg.matmul(matrix, logits)
      logits += self._biases[i]
      if i < len(self.num_filters):
        factor = tf.math.tanh(self._factors[i])
        logits += factor * tf.math.tanh(logits)

    # Convert back to (broadcasted) input tensor shape.
    logits = tf.transpose(logits, (2, 1, 0))
    logits = tf.reshape(logits, shape)
    return logits

  def _log_cdf(self, inputs):
    inputs = self._broadcast_inputs(inputs)
    logits = self._logits_cumulative(inputs)
    return tf.math.log_sigmoid(logits)

  def _log_survival_function(self, inputs):
    inputs = self._broadcast_inputs(inputs)
    logits = self._logits_cumulative(inputs)
    # 1-sigmoid(x) = sigmoid(-x)
    return tf.math.log_sigmoid(-logits)

  def _cdf(self, inputs):
    inputs = self._broadcast_inputs(inputs)
    logits = self._logits_cumulative(inputs)
    return tf.math.sigmoid(logits)

  def _survival_function(self, inputs):
    inputs = self._broadcast_inputs(inputs)
    logits = self._logits_cumulative(inputs)
    # 1-sigmoid(x) = sigmoid(-x)
    return tf.math.sigmoid(-logits)

  def _prob(self, inputs):
    inputs = self._broadcast_inputs(inputs)
    with tf.GradientTape(watch_accessed_variables=False) as tape:
      tape.watch(inputs)
      cdf = self._cdf(inputs)
    prob = tape.gradient(cdf, inputs)
    return prob

  def _log_prob(self, inputs):
    inputs = self._broadcast_inputs(inputs)
    with tf.GradientTape(watch_accessed_variables=False) as tape:
      tape.watch(inputs)
      logits = self._logits_cumulative(inputs)
    # Let x=inputs and s(x)=sigmoid(x).
    # We have F(x) = s(logits(x)),
    # so p(x) = F'(x)
    #         = s'(logits(x)) * logits'(x)
    #         = s(logits(x))*s(-logits(x)) * logits'(x)
    # so log p(x) = log(s(logits(x)) + log(s(-logits(x)) + log(logits'(x)).
    log_s_logits = tf.math.log_sigmoid(logits)
    log_s_neg_logits = tf.math.log_sigmoid(-logits)
    dlogits = tape.gradient(logits, inputs)
    return log_s_logits + log_s_neg_logits + tf.math.log(dlogits)

  def _quantization_offset(self):
    return tf.constant(0, dtype=self.dtype)

  def _lower_tail(self, tail_mass):
    logits = tf.math.log(tail_mass / 2 / (1. - tail_mass / 2))
    return helpers.estimate_tails(
        self._logits_cumulative, logits, self.batch_shape_tensor(), self.dtype)

  def _upper_tail(self, tail_mass):
    logits = -tf.math.log(tail_mass / 2 / (1. - tail_mass / 2))
    return helpers.estimate_tails(
        self._logits_cumulative, logits, self.batch_shape_tensor(), self.dtype)


class NoisyDeepFactorized(uniform_noise.UniformNoiseAdapter):
  """DeepFactorized that is convolved with uniform noise."""

  def __init__(self, name="NoisyDeepFactorized", **kwargs):
    super().__init__(DeepFactorized(**kwargs), name=name)
