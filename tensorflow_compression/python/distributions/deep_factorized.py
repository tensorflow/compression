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

import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp

from tensorflow_compression.python.distributions import helpers
from tensorflow_compression.python.ops import math_ops


__all__ = ["DeepFactorized"]


class DeepFactorized(tfp.distributions.Distribution):
  """Fully factorized distribution based on neural network cumulative.

  This is a flexible, nonparametric probability density model, described in
  appendix 6.1 of the paper:

  > "Variational image compression with a scale hyperprior"<br />
  > J. Ballé, D. Minnen, S. Singh, S. J. Hwang, N. Johnston<br />
  > https://openreview.net/forum?id=rkcQFMZRb

  This implementation already includes convolution with a unit-width uniform
  density, as described in appendix 6.2 of the same paper. Please cite the paper
  if you use this code for scientific work.

  This is a scalar distribution (i.e., its `event_shape` is always length 0),
  and the density object always creates its own `tf.Variable`s representing the
  trainable distribution parameters.
  """

  def __init__(self, batch_shape=(), num_filters=(3, 3), init_scale=10,
               allow_nan_stats=False, dtype=tf.float32, name="DeepFactorized"):
    """Initializer.

    Arguments:
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
    self._estimated_tail_mass = None
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
      init = tf.math.log(tf.math.expm1(1 / scale / filters[i + 1]))
      init = tf.cast(init, dtype=self.dtype)
      init = tf.broadcast_to(init, (channels, filters[i + 1], filters[i]))
      matrix = tf.Variable(init, name="matrix_{}".format(i))
      self._matrices.append(matrix)

      bias = tf.Variable(
          tf.random.uniform(
              (channels, filters[i + 1], 1), -.5, .5, dtype=self.dtype),
          name="bias_{}".format(i))
      self._biases.append(bias)

      if i < len(self.num_filters):
        factor = tf.Variable(
            tf.zeros((channels, filters[i + 1], 1), dtype=self.dtype),
            name="factor_{}".format(i))
        self._factors.append(factor)

  def _batch_shape_tensor(self):
    return tf.constant(self._batch_shape_tuple, dtype=int)

  def _batch_shape(self):
    return tf.TensorShape(self._batch_shape_tuple)

  def _event_shape_tensor(self):
    return tf.constant((), dtype=int)

  def _event_shape(self):
    return tf.TensorShape(())

  def _logits_cumulative(self, inputs):
    """Evaluate logits of the cumulative densities.

    Arguments:
      inputs: The values at which to evaluate the cumulative densities, expected
        to be a `tf.Tensor` of shape `(channels, 1, batch)`.

    Returns:
      A `tf.Tensor` of the same shape as `inputs`, containing the logits of the
      cumulative densities evaluated at the given inputs.
    """
    logits = inputs
    for i in range(len(self.num_filters) + 1):
      matrix = tf.nn.softplus(self._matrices[i])
      logits = tf.linalg.matmul(matrix, logits)
      logits += self._biases[i]
      if i < len(self.num_filters):
        factor = tf.math.tanh(self._factors[i])
        logits += factor * tf.math.tanh(logits)
    return logits

  def _prob(self, y):
    """Called by the base class to compute likelihoods."""
    # Convert to (channels, 1, batch) format by collapsing dimensions and then
    # commuting channels to front.
    y = tf.broadcast_to(
        y, tf.broadcast_dynamic_shape(tf.shape(y), self.batch_shape_tensor()))
    shape = tf.shape(y)
    y = tf.reshape(y, (-1, 1, self.batch_shape.num_elements()))
    y = tf.transpose(y, (2, 1, 0))

    # Evaluate densities.
    # We can use the special rule below to only compute differences in the left
    # tail of the sigmoid. This increases numerical stability: sigmoid(x) is 1
    # for large x, 0 for small x. Subtracting two numbers close to 0 can be done
    # with much higher precision than subtracting two numbers close to 1.
    lower = self._logits_cumulative(y - .5)
    upper = self._logits_cumulative(y + .5)
    # Flip signs if we can move more towards the left tail of the sigmoid.
    sign = tf.stop_gradient(-tf.math.sign(lower + upper))
    p = abs(tf.sigmoid(sign * upper) - tf.sigmoid(sign * lower))
    p = math_ops.lower_bound(p, 0.)

    # Convert back to (broadcasted) input tensor shape.
    p = tf.transpose(p, (2, 1, 0))
    p = tf.reshape(p, shape)
    return p

  def _quantization_offset(self):
    return tf.constant(0, dtype=self.dtype)

  def _lower_tail(self, tail_mass):
    tail = helpers.estimate_tails(
        self._logits_cumulative, -tf.math.log(2 / tail_mass - 1),
        tf.constant([self.batch_shape.num_elements(), 1, 1], tf.int32),
        self.dtype)
    return tf.reshape(tail, self.batch_shape_tensor())

  def _upper_tail(self, tail_mass):
    tail = helpers.estimate_tails(
        self._logits_cumulative, tf.math.log(2 / tail_mass - 1),
        tf.constant([self.batch_shape.num_elements(), 1, 1], tf.int32),
        self.dtype)
    return tf.reshape(tail, self.batch_shape_tensor())
