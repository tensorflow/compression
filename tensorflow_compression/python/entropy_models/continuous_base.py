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
"""Base class for continuous entropy models."""

import abc
import functools

from absl import logging
import tensorflow.compat.v2 as tf

from tensorflow_compression.python.distributions import helpers
from tensorflow_compression.python.ops import range_coding_ops


__all__ = ["ContinuousEntropyModelBase"]


class ContinuousEntropyModelBase(tf.Module, metaclass=abc.ABCMeta):
  """Base class for continuous entropy models.

  The basic functionality of this class is to pre-compute integer probability
  tables based on the provided `tfp.distributions.Distribution` object, which
  can then be used reliably across different platforms by the range coder and
  decoder.
  """

  @abc.abstractmethod
  def __init__(self, prior, coding_rank, compression=False,
               likelihood_bound=1e-9, tail_mass=2**-8,
               range_coder_precision=12, no_variables=False):
    """Initializer.

    Arguments:
      prior: A `tfp.distributions.Distribution` object. A density model fitting
        the marginal distribution of the bottleneck data with additive uniform
        noise, which is shared a priori between the sender and the receiver. For
        best results, the distribution should be flexible enough to have a
        unit-width uniform distribution as a special case, since this is the
        marginal distribution for bottleneck dimensions that are constant.
      coding_rank: Integer. Number of innermost dimensions considered a coding
        unit. Each coding unit is compressed to its own bit string, and the
        `bits()` method sums over each coding unit.
      compression: Boolean. If set to `True`, the range coding tables used by
        `compress()` and `decompress()` will be built on instantiation. If set
        to `False`, these two methods will not be accessible.
      likelihood_bound: Float. Lower bound for likelihood values, to prevent
        training instabilities.
      tail_mass: Float. Approximate probability mass which is range encoded with
        less precision, by using a Golomb-like code.
      range_coder_precision: Integer. Precision passed to the range coding op.
      no_variables: Boolean. If True, creates range coding tables as `Tensor`s
        rather than `Variable`s.

    Raises:
      RuntimeError: when attempting to instantiate an entropy model with
        `compression=True` and not in eager execution mode.
    """
    if prior.event_shape.rank:
      raise ValueError("`prior` must be a (batch of) scalar distribution(s).")
    super().__init__()
    with self.name_scope:
      self._prior = prior
      self._dtype = tf.as_dtype(prior.dtype)
      self._prior_shape = tuple(int(s) for s in prior.batch_shape)
      self._coding_rank = int(coding_rank)
      self._compression = bool(compression)
      self._likelihood_bound = float(likelihood_bound)
      self._tail_mass = float(tail_mass)
      self._range_coder_precision = int(range_coder_precision)
      self._no_variables = bool(no_variables)
      if self.compression:
        self._build_tables(prior)

  @property
  def prior(self):
    """Prior distribution, used for range coding."""
    if not hasattr(self, "_prior"):
      raise RuntimeError(
          "This entropy model doesn't hold a reference to its prior "
          "distribution. This can happen when it is unserialized, because "
          "the prior is not generally serializable.")
    return self._prior

  @prior.deleter
  def prior(self):
    del self._prior

  def _check_compression(self):
    if not self.compression:
      raise RuntimeError(
          "For range coding, the entropy model must be instantiated with "
          "`compression=True`.")

  @property
  def cdf(self):
    self._check_compression()
    return tf.convert_to_tensor(self._cdf)

  @property
  def cdf_offset(self):
    self._check_compression()
    return tf.convert_to_tensor(self._cdf_offset)

  @property
  def cdf_length(self):
    self._check_compression()
    return tf.convert_to_tensor(self._cdf_length)

  @property
  def dtype(self):
    """Data type of this entropy model."""
    return self._dtype

  @property
  def prior_shape(self):
    """Batch shape of `prior` (dimensions which are not assumed i.i.d.)."""
    return self._prior_shape

  @property
  def prior_shape_tensor(self):
    """Batch shape of `prior` as a `Tensor`."""
    return tf.constant(self.prior_shape, dtype=tf.int32)

  @property
  def coding_rank(self):
    """Number of innermost dimensions considered a coding unit."""
    return self._coding_rank

  @property
  def compression(self):
    """Whether this entropy model is prepared for compression."""
    return self._compression

  @property
  def likelihood_bound(self):
    """Lower bound for likelihood values."""
    return self._likelihood_bound

  @property
  def tail_mass(self):
    """Approximate probability mass which is range encoded with overflow."""
    return self._tail_mass

  @property
  def range_coder_precision(self):
    """Precision passed to range coding op."""
    return self._range_coder_precision

  @property
  def no_variables(self):
    """Whether range coding tables are created as `Tensor`s or `Variable`s."""
    return self._no_variables

  @tf.custom_gradient
  def _quantize_no_offset(self, inputs):
    return tf.round(inputs), lambda x: x

  @tf.custom_gradient
  def _quantize_offset(self, inputs, offset):
    return tf.round(inputs - offset) + offset, lambda x: (x, None)

  def _quantize(self, inputs, offset=None):
    if offset is None:
      outputs = self._quantize_no_offset(inputs)
    else:
      outputs = self._quantize_offset(inputs, offset)
    return outputs

  def _build_tables(self, prior):
    """Computes integer-valued probability tables used by the range coder.

    These tables must not be re-generated independently on the sending and
    receiving side, since small numerical discrepancies between both sides can
    occur in this process. If the tables differ slightly, this in turn would
    very likely cause catastrophic error propagation during range decoding. For
    a more in-depth discussion of this, see:

    > "Integer Networks for Data Compression with Latent-Variable Models"<br />
    > J. Ballé, N. Johnston, D. Minnen<br />
    > https://openreview.net/forum?id=S1zz2i0cY7

    The tables are stored in `tf.Variable`s as attributes of this object. The
    recommended way is to train the model, instantiate an entropy model with
    `compression=True`, and then distribute the model to a sender and a
    receiver.

    Arguments:
      prior: The `tfp.distributions.Distribution` object (see initializer).
    """
    offset = helpers.quantization_offset(prior)
    lower_tail = helpers.lower_tail(prior, self.tail_mass)
    upper_tail = helpers.upper_tail(prior, self.tail_mass)

    # Largest distance observed between lower tail and median, and between
    # median and upper tail.
    minima = offset - lower_tail
    minima = tf.cast(tf.math.ceil(minima), tf.int32)
    minima = tf.math.maximum(minima, 0)
    maxima = upper_tail - offset
    maxima = tf.cast(tf.math.ceil(maxima), tf.int32)
    maxima = tf.math.maximum(maxima, 0)

    # PMF starting positions and lengths.
    pmf_start = offset - tf.cast(minima, self.dtype)
    pmf_length = maxima + minima + 1

    # Sample the densities in the computed ranges, possibly computing more
    # samples than necessary at the upper end.
    max_length = tf.math.reduce_max(pmf_length)
    if tf.executing_eagerly() and max_length > 2048:
      logging.warning(
          "Very wide PMF with %d elements may lead to out of memory issues. "
          "Consider priors with smaller dispersion or increasing `tail_mass` "
          "parameter.", int(max_length))
    samples = tf.range(tf.cast(max_length, self.dtype), dtype=self.dtype)
    samples = tf.reshape(samples, [-1] + len(self.prior_shape) * [1])
    samples += pmf_start
    pmf = prior.prob(samples)

    # Collapse batch dimensions of distribution.
    pmf = tf.reshape(pmf, [max_length, -1])
    pmf = tf.transpose(pmf)

    pmf_length = tf.broadcast_to(pmf_length, self.prior_shape_tensor)
    pmf_length = tf.reshape(pmf_length, [-1])
    cdf_length = pmf_length + 2
    cdf_offset = tf.broadcast_to(-minima, self.prior_shape_tensor)
    cdf_offset = tf.reshape(cdf_offset, [-1])

    # Prevent tensors from bouncing back and forth between host and GPU.
    with tf.device("/cpu:0"):
      def loop_body(args):
        prob, length = args
        prob = prob[:length]
        overflow = tf.math.maximum(1 - tf.reduce_sum(prob, keepdims=True), 0.)
        prob = tf.concat([prob, overflow], axis=0)
        cdf = range_coding_ops.pmf_to_quantized_cdf(
            prob, precision=self.range_coder_precision)
        return tf.pad(
            cdf, [[0, max_length - length]], mode="CONSTANT", constant_values=0)

      # TODO(jonycgn,ssjhv): Consider switching to Python control flow.
      cdf = tf.map_fn(
          loop_body, (pmf, pmf_length), dtype=tf.int32, name="pmf_to_cdf")

    if self.no_variables:
      self._cdf = cdf
      self._cdf_offset = cdf_offset
      self._cdf_length = cdf_length
    else:
      self._cdf = tf.Variable(cdf, trainable=False, name="cdf")
      self._cdf_offset = tf.Variable(
          cdf_offset, trainable=False, name="cdf_offset")
      self._cdf_length = tf.Variable(
          cdf_length, trainable=False, name="cdf_length")

  @abc.abstractmethod
  def get_config(self):
    """Returns the configuration of the entropy model.

    Returns:
      A JSON-serializable Python dict.

    Raises:
      NotImplementedError: on attempting to call this method on an entropy model
        with `compression=False`.
    """
    if self.no_variables or not self.compression:
      raise NotImplementedError(
          "Serializing entropy models with `compression=False` or "
          "`no_variables=True` is currently not supported.")
    return dict(
        dtype=self._dtype.name,
        prior_shape=self._prior_shape,
        coding_rank=self._coding_rank,
        likelihood_bound=self._likelihood_bound,
        tail_mass=self._tail_mass,
        range_coder_precision=self._range_coder_precision,
        cdf_width=self._cdf.shape.as_list()[1],
    )

  @classmethod
  @abc.abstractmethod
  def from_config(cls, config):
    """Instantiates an entropy model from a configuration dictionary.

    Arguments:
      config: A `dict`, typically the output of `get_config`.

    Returns:
      An entropy model.
    """
    # Instantiate new object without calling initializers, and call superclass
    # (tf.Module) initializer manually. Note: `cls` is child class of this one.
    self = cls.__new__(cls)  # pylint:disable=no-value-for-parameter
    super().__init__(self)

    # What follows is the alternative initializer.
    with self.name_scope:
      # pylint:disable=protected-access
      self._dtype = tf.as_dtype(config["dtype"])
      self._prior_shape = tuple(int(s) for s in config["prior_shape"])
      self._coding_rank = int(config["coding_rank"])
      self._compression = True
      self._likelihood_bound = float(config["likelihood_bound"])
      self._tail_mass = float(config["tail_mass"])
      self._range_coder_precision = int(config["range_coder_precision"])
      self._no_variables = False

      prior_size = functools.reduce(lambda x, y: x * y, self.prior_shape, 1)
      cdf_width = int(config["cdf_width"])
      zeros = tf.zeros([prior_size, cdf_width], dtype=tf.int32)
      self._cdf = tf.Variable(zeros, trainable=False, name="cdf")
      self._cdf_offset = tf.Variable(
          zeros[:, 0], trainable=False, name="cdf_offset")
      self._cdf_length = tf.Variable(
          zeros[:, 0], trainable=False, name="cdf_length")
      # pylint:enable=protected-access

    return self

  def get_weights(self):
    return tf.keras.backend.batch_get_value(self.variables)

  def set_weights(self, weights):
    if len(weights) != len(self.variables):
      raise ValueError(
          "`set_weights` expects a list of {} arrays, received {}."
          "".format(len(self.variables), len(weights)))
    tf.keras.backend.batch_set_value(zip(self.variables, weights))
