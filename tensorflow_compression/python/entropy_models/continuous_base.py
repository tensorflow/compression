# Lint as: python3
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
  def __init__(self, distribution, coding_rank,
               likelihood_bound=1e-9, tail_mass=2**-8,
               range_coder_precision=12):
    """Initializer.

    Arguments:
      distribution: A `tfp.distributions.Distribution` object modeling the
        distribution of the input data including additive uniform noise. For
        best results, the distribution should be flexible enough to have a
        unit-width uniform distribution as a special case.
      coding_rank: Integer. Number of innermost dimensions considered a coding
        unit. Each coding unit is compressed to its own bit string, and the
        `bits()` method sums over each coding unit.
      likelihood_bound: Float. Lower bound for likelihood values, to prevent
        training instabilities.
      tail_mass: Float. Approximate probability mass which is range encoded with
        less precision, by using a Golomb-like code.
      range_coder_precision: Integer. Precision passed to the range coding op.
    """
    if not distribution.is_scalar_event():
      raise ValueError(
          "`distribution` must be a (batch of) scalar distribution(s).")
    super().__init__()
    self._distribution = distribution
    self._coding_rank = int(coding_rank)
    self._likelihood_bound = float(likelihood_bound)
    self._tail_mass = float(tail_mass)
    self._range_coder_precision = int(range_coder_precision)
    self.update_tables()

  @property
  def distribution(self):
    """Distribution modeling data + i.i.d. uniform noise."""
    return self._distribution

  @property
  def coding_rank(self):
    """Number of innermost dimensions considered a coding unit."""
    return self._coding_rank

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
  def dtype(self):
    """Data type of this distribution."""
    return self.distribution.dtype

  def quantization_offset(self):
    """Distribution-dependent quantization offset."""
    return helpers.quantization_offset(self.distribution)

  def lower_tail(self):
    """Approximate lower tail quantile for range coding."""
    return helpers.lower_tail(self.distribution, self.tail_mass)

  def upper_tail(self):
    """Approximate upper tail quantile for range coding."""
    return helpers.upper_tail(self.distribution, self.tail_mass)

  @tf.custom_gradient
  def _quantize(self, inputs, offset):
    return tf.round(inputs - offset) + offset, lambda x: (x, None)

  def update_tables(self):
    """Updates integer-valued probability tables used by the range coder.

    These tables must not be re-generated independently on the sending and
    receiving side, since small numerical discrepancies between both sides can
    occur in this process. If the tables differ slightly, this in turn would
    very likely cause catastrophic error propagation during range decoding. For
    a more in-depth discussion of this, see:

    > "Integer Networks for Data Compression with Latent-Variable Models"<br />
    > J. Ballé, N. Johnston, D. Minnen<br />
    > https://openreview.net/forum?id=S1zz2i0cY7

    The tables are stored in `tf.Tensor`s as attributes of this object. The
    recommended way is to train the model, then call this method, and then
    distribute the model to a sender and a receiver.
    """
    offset = self.quantization_offset()
    lower_tail = self.lower_tail()
    upper_tail = self.upper_tail()

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
    if max_length > 2048:
      logging.warning(
          "Very wide PMF with %d elements may lead to out of memory issues. "
          "Consider encoding distributions with smaller dispersion or "
          "increasing `tail_mass` parameter.", int(max_length))
    samples = tf.range(tf.cast(max_length, self.dtype), dtype=self.dtype)
    samples = tf.reshape(
        samples, [-1] + self.distribution.batch_shape.rank * [1])
    samples += pmf_start
    pmf = self.distribution.prob(samples)

    # Collapse batch dimensions of distribution.
    pmf = tf.reshape(pmf, [max_length, -1])
    pmf = tf.transpose(pmf)

    dist_shape = self.distribution.batch_shape_tensor()
    pmf_length = tf.broadcast_to(pmf_length, dist_shape)
    pmf_length = tf.reshape(pmf_length, [-1])
    cdf_length = pmf_length + 2
    cdf_offset = tf.broadcast_to(-minima, dist_shape)
    cdf_offset = tf.reshape(cdf_offset, [-1])

    # Prevent tensors from bouncing back and forth between host and GPU.
    with tf.device("/cpu:0"):
      def loop_body(args):
        prob, length = args
        prob = prob[:length]
        prob = tf.concat([prob, 1 - tf.reduce_sum(prob, keepdims=True)], axis=0)
        cdf = range_coding_ops.pmf_to_quantized_cdf(
            prob, precision=self.range_coder_precision)
        return tf.pad(
            cdf, [[0, max_length - length]], mode="CONSTANT", constant_values=0)

      # TODO(jonycgn,ssjhv): Consider switching to Python control flow.
      cdf = tf.map_fn(
          loop_body, (pmf, pmf_length), dtype=tf.int32, name="pmf_to_cdf")

    self._cdf, self._cdf_offset, self._cdf_length = cdf, cdf_offset, cdf_length
