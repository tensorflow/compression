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

import functools
import tensorflow as tf
from tensorflow_compression.python.distributions import helpers
from tensorflow_compression.python.entropy_models import continuous_base
from tensorflow_compression.python.ops import math_ops
from tensorflow_compression.python.ops import range_coding_ops


__all__ = [
    "ContinuousBatchedEntropyModel",
]


@tf.keras.utils.register_keras_serializable(package="tensorflow_compression")
class ContinuousBatchedEntropyModel(continuous_base.ContinuousEntropyModelBase):
  """Batched entropy model for continuous random variables.

  This entropy model handles quantization of a bottleneck tensor and helps with
  training of the parameters of the probability distribution modeling the
  tensor (a shared "prior" between sender and receiver). It also pre-computes
  integer probability tables, which can then be used to compress and decompress
  bottleneck tensors reliably across different platforms.

  A typical workflow looks like this:

  - Train a model using an instance of this entropy model as a bottleneck,
    passing the bottleneck tensor through it. With training=True, the model
    computes a differentiable upper bound on the number of bits needed to
    compress the bottleneck tensor.
  - For evaluation, get a closer estimate of the number of compressed bits
    using training=False`.
  - Instantiate an entropy model with `compression=True` (and the same
    parameters as during training), and share the model between a sender and a
    receiver.
  - On the sender side, compute the bottleneck tensor and call `compress()` on
    it. The output is a compressed string representation of the tensor. Transmit
    the string to the receiver, and call `decompress()` there. The output is the
    quantized bottleneck tensor. Continue processing the tensor on the receiving
    side.

  This class assumes that all scalar elements of the encoded tensor are
  statistically independent, and that the parameters of their scalar
  distributions do not depend on data. The innermost dimensions of the
  bottleneck tensor must be broadcastable to the batch shape of `prior`. Any
  dimensions to the left of the batch shape are assumed to be i.i.d., i.e. the
  likelihoods are broadcast to the bottleneck tensor accordingly.

  A more detailed description (and motivation) of this way of performing
  quantization and range coding can be found in the following paper. Please cite
  the paper when using this code for derivative work.

  > "End-to-end Optimized Image Compression"<br />
  > J. Ballé, V. Laparra, E.P. Simoncelli<br />
  > https://openreview.net/forum?id=rJxdQ3jeg
  """

  def __init__(self,
               prior,
               coding_rank,
               compression=False,
               laplace_tail_mass=0.0,
               expected_grads=False,
               tail_mass=2**-8,
               range_coder_precision=12,
               no_variables=False):
    """Initializes the instance.

    Args:
      prior: A `tfp.distributions.Distribution` object. A density model fitting
        the marginal distribution of the bottleneck data with additive uniform
        noise, which is shared a priori between the sender and the receiver. For
        best results, the distribution should be flexible enough to have a
        unit-width uniform distribution as a special case, since this is the
        marginal distribution for bottleneck dimensions that are constant. The
        distribution parameters may not depend on data (they must be either
        variables or constants).
      coding_rank: Integer. Number of innermost dimensions considered a coding
        unit. Each coding unit is compressed to its own bit string, and the
        bits in the __call__ method are summed over each coding unit.
      compression: Boolean. If set to `True`, the range coding tables used by
        `compress()` and `decompress()` will be built on instantiation. If set
        to `False`, these two methods will not be accessible.
      laplace_tail_mass: Float. If positive, will augment the prior with a
        Laplace mixture for training stability. (experimental)
      expected_grads: If True, will use analytical expected gradients during
        backpropagation w.r.t. additive uniform noise.
      tail_mass: Float. Approximate probability mass which is range encoded with
        less precision, by using a Golomb-like code.
      range_coder_precision: Integer. Precision passed to the range coding op.
      no_variables: Boolean. If True, creates range coding tables as `Tensor`s
        rather than `Variable`s.

    Raises:
      RuntimeError: when attempting to instantiate an entropy model with
        `compression=True` and not in eager execution mode.
    """
    if coding_rank < prior.batch_shape.rank:
      raise ValueError(
          "`coding_rank` can't be smaller than batch rank of prior.")
    super().__init__(
        prior=prior,
        coding_rank=coding_rank,
        compression=compression,
        laplace_tail_mass=laplace_tail_mass,
        expected_grads=expected_grads,
        tail_mass=tail_mass,
        range_coder_precision=range_coder_precision,
        no_variables=no_variables,
    )
    self._cache_quantization_offset()

  def _cache_quantization_offset(self):
    """Comptue quantization offset from prior and cache it."""
    quantization_offset = helpers.quantization_offset(self.prior)
    # Optimization: if the quantization offset is zero, we don't need to
    # subtract/add it when quantizing, and we don't need to serialize its value.
    # Note that this code will only work in eager mode.
    # TODO(jonycgn): Reconsider if this optimization is worth keeping once the
    # implementation is stable.
    if tf.executing_eagerly() and tf.reduce_all(
        tf.equal(quantization_offset, 0.)):
      quantization_offset = None
    else:
      quantization_offset = tf.broadcast_to(
          quantization_offset, self.prior_shape_tensor)
      if self.compression and not self.no_variables:
        quantization_offset = tf.Variable(
            quantization_offset, trainable=False, name="quantization_offset")
    self._quantization_offset = quantization_offset

  @property
  def quantization_offset(self):
    if self._quantization_offset is None:
      return None
    return tf.convert_to_tensor(self._quantization_offset)

  def _compute_indexes_and_offset(self, broadcast_shape):
    """Returns the indexes for range coding and the quantization offset."""
    # TODO(jonycgn, ssjhv): Investigate broadcasting in range coding op.
    prior_size = functools.reduce(lambda x, y: x * y, self.prior_shape, 1)
    indexes = tf.range(prior_size, dtype=tf.int32)
    indexes = tf.reshape(indexes, self.prior_shape_tensor)
    indexes = tf.broadcast_to(
        indexes, tf.concat([broadcast_shape, self.prior_shape_tensor], 0))
    return indexes, self.quantization_offset

  @tf.Module.with_name_scope
  def __call__(self, bottleneck, training=True):
    """Perturbs a tensor with (quantization) noise and estimates bitcost.

    Args:
      bottleneck: `tf.Tensor` containing the data to be compressed. Must have at
        least `self.coding_rank` dimensions, and the innermost dimensions must
        be broadcastable to `self.prior_shape`.
      training: Boolean. If `False`, computes the Shannon information of
        `bottleneck` under the distribution `self.prior`, which is a
        non-differentiable, tight *lower* bound on the number of bits needed to
        compress `bottleneck` using `compress()`. If `True`, returns a somewhat
        looser, but differentiable *upper* bound on this quantity.

    Returns:
      A tuple (bottleneck_perturbed, bits) where `bottleneck_perturbed` is
      `bottleneck` perturbed with (quantization) noise, and `bits` is the
      bitcost with the same shape as `bottleneck` without the `self.coding_rank`
      innermost dimensions.
    """
    log_prob_fn = functools.partial(self._log_prob_from_prior, self.prior)
    if training:
      log_probs, bottleneck_perturbed = math_ops.perturb_and_apply(
          log_prob_fn, bottleneck, expected_grads=self._expected_grads)
    else:
      bottleneck_perturbed = self.quantize(bottleneck)
      log_probs = log_prob_fn(bottleneck_perturbed)

    axes = tuple(range(-self.coding_rank, 0))
    bits = tf.reduce_sum(log_probs, axis=axes) / (
        -tf.math.log(tf.constant(2, dtype=log_probs.dtype)))
    return bottleneck_perturbed, bits

  @tf.Module.with_name_scope
  def quantize(self, bottleneck):
    """Quantizes a floating-point tensor.

    To use this entropy model as an information bottleneck during training, pass
    a tensor through this function. The tensor is rounded to integer values
    shifted by an offset, which depends on `self.prior`. For instance, for a
    Gaussian distribution, the returned values are rounded to the location of
    the mode of the distribution plus or minus an integer.

    The gradient of this rounding operation is overridden with the identity
    (straight-through gradient estimator).

    Args:
      bottleneck: `tf.Tensor` containing the data to be quantized. The innermost
        dimensions must be broadcastable to `self.prior_shape`.

    Returns:
      A `tf.Tensor` containing the quantized values.
    """
    return self._quantize(bottleneck, self.quantization_offset)

  @tf.Module.with_name_scope
  def compress(self, bottleneck):
    """Compresses a floating-point tensor.

    Compresses the tensor to bit strings. `bottleneck` is first quantized
    as in `quantize()`, and then compressed using the probability tables derived
    from `self.prior`. The quantized tensor can later be recovered by
    calling `decompress()`.

    The innermost `self.coding_rank` dimensions are treated as one coding unit,
    i.e. are compressed into one string each. Any additional dimensions to the
    left are treated as batch dimensions.

    Args:
      bottleneck: `tf.Tensor` containing the data to be compressed. Must have at
        least `self.coding_rank` dimensions, and the innermost dimensions must
        be broadcastable to `self.prior_shape`.

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
        :self.coding_rank - len(self.prior_shape)]

    indexes, offset = self._compute_indexes_and_offset(broadcast_shape)
    if offset is not None:
      bottleneck -= offset
    symbols = tf.cast(tf.round(bottleneck), tf.int32)
    symbols = tf.reshape(symbols, tf.concat([[-1], coding_shape], 0))

    # Prevent tensors from bouncing back and forth between host and GPU.
    with tf.device("/cpu:0"):
      cdf = self.cdf
      cdf_length = self.cdf_length
      cdf_offset = self.cdf_offset
      def loop_body(symbols):
        return range_coding_ops.unbounded_index_range_encode(
            symbols, indexes, cdf, cdf_length, cdf_offset,
            precision=self.range_coder_precision,
            overflow_width=4, debug_level=1)

      # TODO(jonycgn,ssjhv): Consider switching to Python control flow.
      strings = tf.map_fn(
          loop_body, symbols, dtype=tf.string, name="compress")

    strings = tf.reshape(strings, batch_shape)
    return strings

  @tf.Module.with_name_scope
  def decompress(self, strings, broadcast_shape):
    """Decompresses a tensor.

    Reconstructs the quantized tensor from bit strings produced by `compress()`.
    It is necessary to provide a part of the output shape in `broadcast_shape`.

    Args:
      strings: `tf.Tensor` containing the compressed bit strings.
      broadcast_shape: Iterable of ints. The part of the output tensor shape
        between the shape of `strings` on the left and `self.prior_shape` on the
        right. This must match the shape of the input to `compress()`.

    Returns:
      A `tf.Tensor` of shape `strings.shape + broadcast_shape +
      self.prior_shape`.
    """
    strings = tf.convert_to_tensor(strings, dtype=tf.string)
    broadcast_shape = tf.convert_to_tensor(broadcast_shape, dtype=tf.int32)
    batch_shape = tf.shape(strings)
    symbols_shape = tf.concat(
        [batch_shape, broadcast_shape, self.prior_shape_tensor], 0)

    indexes, offset = self._compute_indexes_and_offset(broadcast_shape)
    strings = tf.reshape(strings, [-1])

    # Prevent tensors from bouncing back and forth between host and GPU.
    with tf.device("/cpu:0"):
      cdf = self.cdf
      cdf_length = self.cdf_length
      cdf_offset = self.cdf_offset
      def loop_body(string):
        return range_coding_ops.unbounded_index_range_decode(
            string, indexes, cdf, cdf_length, cdf_offset,
            precision=self.range_coder_precision,
            overflow_width=4, debug_level=1)

      # TODO(jonycgn,ssjhv): Consider switching to Python control flow.
      symbols = tf.map_fn(
          loop_body, strings, dtype=tf.int32, name="decompress")

    symbols = tf.reshape(symbols, symbols_shape)
    outputs = tf.cast(symbols, self.dtype)
    return outputs + offset if offset is not None else outputs

  def get_config(self):
    """Returns the configuration of the entropy model.

    Returns:
      A JSON-serializable Python dict.

    Raises:
      NotImplementedError: on attempting to call this method on an entropy model
        with `compression=False`.
    """
    config = super().get_config()
    config.update(
        quantization_offset=self._quantization_offset is not None,
    )
    return config

  @classmethod
  def from_config(cls, config):
    """Instantiates an entropy model from a configuration dictionary.

    Args:
      config: A `dict`, typically the output of `get_config`.

    Returns:
      An entropy model.
    """
    self = super().from_config(config)
    with self.name_scope:
      # pylint:disable=protected-access
      if config["quantization_offset"]:
        zeros = tf.zeros(self.prior_shape_tensor, dtype=self.dtype)
        self._quantization_offset = tf.Variable(
            zeros, name="quantization_offset")
      else:
        self._quantization_offset = None
      # pylint:enable=protected-access
    return self
