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
"""Batched entropy model for continuous random variables."""

import tensorflow.compat.v2 as tf

from tensorflow_compression.python.entropy_models import continuous_base
from tensorflow_compression.python.ops import math_ops
from tensorflow_compression.python.ops import range_coding_ops


__all__ = [
    "ContinuousBatchedEntropyModel",
]


class ContinuousBatchedEntropyModel(continuous_base.ContinuousEntropyModelBase):
  """Batched entropy model for continuous random variables.

  This entropy model handles quantization of a bottleneck tensor and helps with
  training of the parameters of the probability distribution modeling the
  tensor. It also pre-computes integer probability tables, which can then be
  used to compress and decompress bottleneck tensors reliably across different
  platforms.

  A typical workflow looks like this:

  - Train a model using this entropy model as a bottleneck, passing the
    bottleneck tensor through `quantize()` while optimizing compressibility of
    the tensor using `bits()`. `bits(training=True)` computes a differentiable
    upper bound on the number of bits needed to compress the bottleneck tensor.
  - For evaluation, get a closer estimate of the number of compressed bits
    using `bits(training=False)`.
  - Call `update_tables()` to ensure the probability tables for range coding are
    up-to-date.
  - Share the model between a sender and a receiver.
  - On the sender side, compute the bottleneck tensor and call `compress()` on
    it. The output is a compressed string representation of the tensor. Transmit
    the string to the receiver, and call `decompress()` there. The output is the
    quantized bottleneck tensor. Continue processing the tensor on the receiving
    side.

  This class assumes that all scalar elements of the encoded tensor are
  statistically independent, and that the parameters of their scalar
  distributions do not depend on data. The innermost dimensions of the
  bottleneck tensor must be broadcastable to the batch shape of `distribution`.
  Any dimensions to the left of the batch shape are assumed to be i.i.d., i.e.
  the likelihoods are broadcast to the bottleneck tensor accordingly.

  A more detailed description (and motivation) of this way of performing
  quantization and range coding can be found in the following paper. Please cite
  the paper when using this code for derivative work.

  > "End-to-end Optimized Image Compression"<br />
  > J. Ballé, V. Laparra, E.P. Simoncelli<br />
  > https://openreview.net/forum?id=rJxdQ3jeg
  """

  def __init__(self, distribution, coding_rank,
               likelihood_bound=1e-9, tail_mass=2**-8,
               range_coder_precision=12):
    """Initializer.

    Arguments:
      distribution: A `tfp.distributions.Distribution` object modeling the
        distribution of the bottleneck tensor values including additive uniform
        noise. The distribution parameters may not depend on data (they must be
        trainable variables or constants). For best results, the distribution
        should be flexible enough to have a unit-width uniform distribution as a
        special case, since this is the distribution an element will take on
        when its bottleneck value is constant (due to the additive noise).
      coding_rank: Integer. Number of innermost dimensions considered a coding
        unit. Each coding unit is compressed to its own bit string, and the
        `bits()` method sums over each coding unit.
      likelihood_bound: Float. Lower bound for likelihood values, to prevent
        training instabilities.
      tail_mass: Float. Approximate probability mass which is range encoded with
        less precision, by using a Golomb-like code.
      range_coder_precision: Integer. Precision passed to the range coding op.
    """
    if coding_rank < distribution.batch_shape.rank:
      raise ValueError(
          "`coding_rank` can't be smaller than batch rank of `distribution`.")
    super().__init__(
        distribution, coding_rank, likelihood_bound=likelihood_bound,
        tail_mass=tail_mass, range_coder_precision=range_coder_precision)

  def _compute_indexes(self, broadcast_shape):
    # TODO(jonycgn, ssjhv): Investigate broadcasting in range coding op.
    dist_shape = self.distribution.batch_shape_tensor()
    indexes = tf.range(tf.reduce_prod(dist_shape), dtype=tf.int32)
    indexes = tf.reshape(indexes, dist_shape)
    indexes = tf.broadcast_to(
        indexes, tf.concat([broadcast_shape, dist_shape], 0))
    return indexes

  def bits(self, bottleneck, training=True):
    """Estimates the number of bits needed to compress a tensor.

    Arguments:
      bottleneck: `tf.Tensor` containing the data to be compressed. Must have at
        least `self.coding_rank` dimensions, and the innermost dimensions must
        be broadcastable to `self.distribution.batch_shape`.
      training: Boolean. If `False`, computes the Shannon information of
        `bottleneck` under the distribution `self.distribution`, which is a
        non-differentiable, tight *lower* bound on the number of bits needed to
        compress `bottleneck` using `compress()`. If `True`, returns a somewhat
        looser, but differentiable *upper* bound on this quantity.

    Returns:
      A `tf.Tensor` having the same shape as `bottleneck` without the
      `self.coding_rank` innermost dimensions, containing the number of bits.
    """
    if training:
      quantized = bottleneck + tf.random.uniform(
          tf.shape(bottleneck), minval=-.5, maxval=.5, dtype=bottleneck.dtype)
    else:
      quantized = self.quantize(bottleneck)
    probs = self.distribution.prob(quantized)
    probs = math_ops.lower_bound(probs, self.likelihood_bound)
    axes = tuple(range(-self.coding_rank, 0))
    bits = tf.reduce_sum(tf.math.log(probs), axis=axes) / -tf.math.log(2.)
    return bits

  def quantize(self, bottleneck):
    """Quantizes a floating-point tensor.

    To use this entropy model as an information bottleneck during training, pass
    a tensor through this function. The tensor is rounded to integer values
    modulo `self.quantization_offset`, which depends on `self.distribution`. For
    instance, for a Gaussian distribution, the returned values are rounded to
    the location of the mode of the distribution plus or minus an integer.

    The gradient of this rounding operation is overridden with the identity
    (straight-through gradient estimator).

    Arguments:
      bottleneck: `tf.Tensor` containing the data to be quantized. The innermost
        dimensions must be broadcastable to `self.distribution.batch_shape`.

    Returns:
      A `tf.Tensor` containing the quantized values.
    """
    offset = self.quantization_offset()
    return self._quantize(bottleneck, offset)

  def compress(self, bottleneck):
    """Compresses a floating-point tensor.

    Compresses the tensor to bit strings. `bottleneck` is first quantized
    as in `quantize()`, and then compressed using the probability tables derived
    from `self.distribution`. The quantized tensor can later be recovered by
    calling `decompress()`.

    The innermost `self.coding_rank` dimensions are treated as one coding unit,
    i.e. are compressed into one string each. Any additional dimensions to the
    left are treated as batch dimensions.

    Arguments:
      bottleneck: `tf.Tensor` containing the data to be compressed. Must have at
        least `self.coding_rank` dimensions, and the innermost dimensions must
        be broadcastable to `self.distribution.batch_shape`.

    Returns:
      A `tf.Tensor` having the same shape as `bottleneck` without the
      `self.coding_rank` innermost dimensions, containing a string for each
      coding unit.
    """
    input_shape = tf.shape(bottleneck)
    input_rank = tf.shape(input_shape)[0]
    batch_shape, coding_shape = tf.split(
        input_shape, [input_rank - self.coding_rank, self.coding_rank])
    broadcast_shape = coding_shape[
        :self.coding_rank - self.distribution.batch_shape.rank]

    indexes = self._compute_indexes(broadcast_shape)
    offset = self.quantization_offset()
    symbols = tf.cast(tf.round(bottleneck - offset), tf.int32)
    symbols = tf.reshape(symbols, tf.concat([[-1], coding_shape], 0))

    # Prevent tensors from bouncing back and forth between host and GPU.
    with tf.device("/cpu:0"):
      def loop_body(symbols):
        return range_coding_ops.unbounded_index_range_encode(
            symbols, indexes,
            self._cdf, self._cdf_length, self._cdf_offset,
            precision=self.range_coder_precision,
            overflow_width=4, debug_level=1)

      # TODO(jonycgn,ssjhv): Consider switching to Python control flow.
      strings = tf.map_fn(
          loop_body, symbols, dtype=tf.string, name="compress")

    strings = tf.reshape(strings, batch_shape)
    return strings

  def decompress(self, strings, broadcast_shape):
    """Decompresses a tensor.

    Reconstructs the quantized tensor from bit strings produced by `compress()`.
    It is necessary to provide a part of the output shape in `broadcast_shape`.

    Arguments:
      strings: `tf.Tensor` containing the compressed bit strings.
      broadcast_shape: Iterable of ints. The part of the output tensor shape
        between the shape of `strings` on the left and
        `self.distribution.batch_shape` on the right. This must match the shape
        of the input to `compress()`.

    Returns:
      A `tf.Tensor` of shape `strings.shape + broadcast_shape +
      self.distribution.batch_shape`.
    """
    batch_shape = tf.shape(strings)
    dist_shape = self.distribution.batch_shape_tensor()
    symbols_shape = tf.concat([batch_shape, broadcast_shape, dist_shape], 0)

    indexes = self._compute_indexes(broadcast_shape)
    strings = tf.reshape(strings, [-1])

    # Prevent tensors from bouncing back and forth between host and GPU.
    with tf.device("/cpu:0"):
      def loop_body(string):
        return range_coding_ops.unbounded_index_range_decode(
            string, indexes,
            self._cdf, self._cdf_length, self._cdf_offset,
            precision=self.range_coder_precision,
            overflow_width=4, debug_level=1)

      # TODO(jonycgn,ssjhv): Consider switching to Python control flow.
      symbols = tf.map_fn(
          loop_body, strings, dtype=tf.int32, name="decompress")

    symbols = tf.reshape(symbols, symbols_shape)
    offset = self.quantization_offset()
    return tf.cast(symbols, self.dtype) + offset
