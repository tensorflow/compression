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
from tensorflow_compression.python.ops import gen_ops
from tensorflow_compression.python.ops import math_ops
from tensorflow_compression.python.ops import round_ops


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
    passing the bottleneck tensor through it. With `training=True`, the model
    computes a differentiable upper bound on the number of bits needed to
    compress the bottleneck tensor.
  - For evaluation, get a closer estimate of the number of compressed bits
    using `training=False`.
  - Instantiate an entropy model with `compression=True` (and the same
    parameters as during training), and share the model between a sender and a
    receiver.
  - On the sender side, compute the bottleneck tensor and call `compress()` on
    it. The output is a compressed string representation of the tensor. Transmit
    the string to the receiver, and call `decompress()` there. The output is the
    quantized bottleneck tensor. Continue processing the tensor on the receiving
    side.

  Entropy models which contain range coding tables (i.e. with
  `compression=True`) can be instantiated in three ways:

  - By providing a continuous "prior" distribution object. The range coding
    tables are then derived from that continuous distribution.
  - From a config as returned by `get_config`, followed by a call to
    `set_weights`. This implements the Keras serialization protocol. In this
    case, the initializer creates empty state variables for the range coding
    tables, which are then filled by `set_weights`. As a consequence, this
    method requires `stateless=False`.
  - In a more low-level way, by directly providing the range coding tables to
    `__init__`, for use cases where the Keras protocol can't be used (e.g., when
    the entropy model must not create variables).

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
               prior=None,
               coding_rank=None,
               compression=False,
               stateless=False,
               expected_grads=False,
               tail_mass=2**-8,
               range_coder_precision=12,
               dtype=None,
               prior_shape=None,
               cdf=None,
               cdf_offset=None,
               cdf_shapes=None,
               non_integer_offset=True,
               quantization_offset=None,
               laplace_tail_mass=0):
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
      stateless: Boolean. If `False`, range coding tables are created as
        `Variable`s. This allows the entropy model to be serialized using the
        `SavedModel` protocol, so that both the encoder and the decoder use
        identical tables when loading the stored model. If `True`, creates range
        coding tables as `Tensor`s. This makes the entropy model stateless and
        allows it to be constructed within a `tf.function` body, for when the
        range coding tables are provided manually. If `compression=False`, then
        `stateless=True` is implied and the provided value is ignored.
      expected_grads: If True, will use analytical expected gradients during
        backpropagation w.r.t. additive uniform noise.
      tail_mass: Float. Approximate probability mass which is encoded using an
        Elias gamma code embedded into the range coder.
      range_coder_precision: Integer. Precision passed to the range coding op.
      dtype: `tf.dtypes.DType`. Data type of this entropy model (i.e. dtype of
        prior, decompressed values). Must be provided if `prior` is omitted.
      prior_shape: Batch shape of the prior (dimensions which are not assumed
        i.i.d.). Must be provided if `prior` is omitted.
      cdf: `tf.Tensor` or `None`. If provided, is used for range coding rather
        than tables built from the prior.
      cdf_offset: `tf.Tensor` or `None`. Must be provided along with `cdf`.
      cdf_shapes: Shapes of `cdf` and `cdf_offset`. If provided, empty range
        coding tables are created, which can then be restored using
        `set_weights`. Requires `compression=True` and `stateless=False`.
      non_integer_offset: Boolean. Whether to quantize to non-integer offsets
        heuristically determined from mode/median of prior. Set this to `False`
        if you are using soft quantization during training.
      quantization_offset: `tf.Tensor` or `None`. If `cdf` is provided and
        `non_integer_offset=True`, must be provided as well.
      laplace_tail_mass: Float. If positive, will augment the prior with a
        Laplace mixture for training stability. (experimental)
    """
    if not (prior is not None) == (dtype is None) == (prior_shape is None):
      raise ValueError(
          "Either `prior` or both `dtype` and `prior_shape` must be provided.")
    if (prior is None) + (cdf_shapes is None) + (cdf is None) != 2:
      raise ValueError(
          "Must provide exactly one of `prior`, `cdf`, or `cdf_shapes`.")
    if not compression and not (
        cdf is None and cdf_offset is None and cdf_shapes is None):
      raise ValueError("CDFs can't be provided with `compression=False`")
    if prior is not None and prior.event_shape.rank:
      raise ValueError("`prior` must be a (batch of) scalar distribution(s).")

    super().__init__(
        coding_rank=coding_rank,
        compression=compression,
        stateless=stateless,
        expected_grads=expected_grads,
        tail_mass=tail_mass,
        dtype=dtype if dtype is not None else prior.dtype,
        laplace_tail_mass=laplace_tail_mass,
    )
    self._prior = prior
    self._non_integer_offset = bool(non_integer_offset)
    self._prior_shape = tf.TensorShape(
        prior_shape if prior is None else prior.batch_shape)
    if self.coding_rank < self.prior_shape.rank:
      raise ValueError("`coding_rank` can't be smaller than `prior_shape`.")

    with self.name_scope:
      if quantization_offset is not None:
        # If quantization offset is passed in manually, use it.
        pass
      elif not self.non_integer_offset:
        # If not using the offset heuristic, always quantize to integers.
        quantization_offset = None
      elif cdf_shapes is not None:
        # `cdf_shapes` being set indicates that we are using the `SavedModel`
        # protocol. So create a placeholder value.
        quantization_offset = tf.zeros(
            self.prior_shape_tensor, dtype=self.dtype)
      elif cdf is not None:
        # CDF is passed in manually. So assume the same about the offsets.
        if quantization_offset is None:
          raise ValueError(
              "When providing `cdf` and `non_integer_offset=True`, must also "
              "provide `quantization_offset`.")
      else:
        assert self._prior is not None
        # If prior is available, determine offsets from it using the heuristic.
        quantization_offset = helpers.quantization_offset(self.prior)
        # Optimization: if the quantization offset is zero, we don't need to
        # subtract/add it when quantizing, and we don't need to serialize its
        # value. Note that this code will only work in eager mode.
        if (tf.executing_eagerly() and
            tf.reduce_all(tf.equal(quantization_offset, 0.))):
          quantization_offset = None
        else:
          quantization_offset = tf.broadcast_to(
              quantization_offset, self.prior_shape_tensor)
      if quantization_offset is None:
        self._quantization_offset = None
      elif self.compression and not self.stateless:
        self._quantization_offset = tf.Variable(
            quantization_offset, dtype=self.dtype, trainable=False,
            name="quantization_offset")
      else:
        self._quantization_offset = tf.convert_to_tensor(
            quantization_offset, dtype=self.dtype, name="quantization_offset")
      if self.compression:
        if cdf is None and cdf_shapes is None:
          cdf, cdf_offset = self._build_tables(
              self.prior, range_coder_precision, offset=quantization_offset)
        self._init_compression(cdf, cdf_offset, cdf_shapes)

  @property
  def prior_shape(self):
    """Batch shape of `prior` (dimensions which are not assumed i.i.d.)."""
    return self._prior_shape

  @property
  def prior_shape_tensor(self):
    """Batch shape of `prior` as a `Tensor`."""
    return tf.constant(self.prior_shape.as_list(), dtype=tf.int32)

  @property
  def non_integer_offset(self):
    return self._non_integer_offset

  @property
  def quantization_offset(self):
    if self._quantization_offset is None:
      return None
    return tf.convert_to_tensor(self._quantization_offset)

  @tf.Module.with_name_scope
  def __call__(self, bottleneck, training=True):
    """Perturbs a tensor with (quantization) noise and estimates rate.

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
      `bottleneck` perturbed with (quantization) noise, and `bits` is the rate.
      `bits` has the same shape as `bottleneck` without the `self.coding_rank`
      innermost dimensions.
    """
    log_prob_fn = functools.partial(self._log_prob, self.prior)
    if training:
      log_probs, bottleneck_perturbed = math_ops.perturb_and_apply(
          log_prob_fn, bottleneck, expected_grads=self.expected_grads)
    else:
      bottleneck_perturbed = self.quantize(bottleneck)
      log_probs = log_prob_fn(bottleneck_perturbed)
    axes = tuple(range(-self.coding_rank, 0))
    bits = tf.reduce_sum(log_probs, axis=axes) / (
        -tf.math.log(tf.constant(2, dtype=log_probs.dtype)))
    return bottleneck_perturbed, bits

  @tf.Module.with_name_scope
  def quantize(self, bottleneck):
    """Quantizes a floating-point bottleneck tensor.

    The tensor is rounded to integer values potentially shifted by offsets (if
    `self.quantization_offset is not None`). These offsets can depend on
    `self.prior`. For instance, for a Gaussian distribution, when
    `self.non_integer_offset == True`, the returned values would be rounded
    to the location of the mode of the distribution plus or minus an integer.

    The gradient of this rounding operation is overridden with the identity
    (straight-through gradient estimator).

    Args:
      bottleneck: `tf.Tensor` containing the data to be quantized. The innermost
        dimensions must be broadcastable to `self.prior_shape`.

    Returns:
      A `tf.Tensor` containing the quantized values.
    """
    return round_ops.round_st(bottleneck, self.quantization_offset)

  @tf.Module.with_name_scope
  def compress(self, bottleneck):
    """Compresses a floating-point tensor.

    Compresses the tensor to bit strings. `bottleneck` is first quantized
    as in `quantize()`, and then compressed using the probability tables in
    `self.cdf` (derived from `self.prior`). The quantized tensor can later be
    recovered by calling `decompress()`.

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
    all_but_last_n_elems = lambda t, n: t[:-n] if n else t
    batch_shape = all_but_last_n_elems(input_shape, self.coding_rank)
    iid_shape = all_but_last_n_elems(input_shape, self.prior_shape.rank)
    offset = self.quantization_offset
    if offset is not None:
      bottleneck -= offset
    symbols = tf.cast(tf.round(bottleneck), tf.int32)
    symbols = tf.reshape(symbols, tf.concat([iid_shape, [-1]], 0))
    symbols -= self.cdf_offset
    handle = gen_ops.create_range_encoder(batch_shape, self.cdf)
    handle = gen_ops.entropy_encode_channel(handle, symbols)
    return gen_ops.entropy_encode_finalize(handle)

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
    decode_shape = tf.concat(
        [broadcast_shape, [tf.reduce_prod(self.prior_shape_tensor)]], 0)
    output_shape = tf.concat(
        [tf.shape(strings), broadcast_shape, self.prior_shape_tensor], 0)
    handle = gen_ops.create_range_decoder(strings, self.cdf)
    handle, symbols = gen_ops.entropy_decode_channel(
        handle, decode_shape, self.cdf_offset.dtype)
    sanity = gen_ops.entropy_decode_finalize(handle)
    tf.debugging.assert_equal(sanity, True, message="Sanity check failed.")
    symbols += self.cdf_offset
    symbols = tf.reshape(symbols, output_shape)
    outputs = tf.cast(symbols, self.dtype)
    offset = self.quantization_offset
    if offset is not None:
      outputs += offset
    return outputs

  def get_config(self):
    """Returns the configuration of the entropy model.

    Returns:
      A JSON-serializable Python dict.
    """
    config = super().get_config()
    config.update(
        prior_shape=tuple(map(int, self.prior_shape)),
        # Since the prior is never passed when using the `SavedModel` protocol,
        # we can reuse this flag to indicate whether the offsets need to be
        # loaded from a variable.
        non_integer_offset=self.quantization_offset is not None,
    )
    return config
