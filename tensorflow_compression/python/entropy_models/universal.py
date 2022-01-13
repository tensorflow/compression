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
"""Entropy models which implement universal quantization."""

import functools
import tensorflow as tf
from tensorflow_compression.python.entropy_models import continuous_base
from tensorflow_compression.python.ops import gen_ops
from tensorflow_compression.python.ops import math_ops


__all__ = [
    "UniversalBatchedEntropyModel",
    "UniversalIndexedEntropyModel",
]


def _add_offset_indexes(indexes, num_noise_levels):
  """Adds offset indexes to `indexes`."""
  # This works as a shared source of randomness across CPU and GPU.
  shape = tf.shape(indexes)[:-1]
  offset_indexes = tf.random.stateless_uniform(
      shape,
      seed=(1234, 1234),
      minval=0,
      maxval=num_noise_levels,
      dtype=tf.int32)
  offset_indexes = tf.cast(offset_indexes, indexes.dtype)
  return tf.concat((offset_indexes[..., None], indexes), axis=-1)


def _offset_indexes_to_offset(offset_indexes,
                              num_noise_levels,
                              dtype=tf.float32):
  return tf.cast(
      (offset_indexes + 1) / (num_noise_levels + 1) - 0.5, dtype=dtype)


def _index_ranges_without_offsets(index_ranges_with_offsets):
  """Return index_ranges excluding the offset dimension."""
  return index_ranges_with_offsets[1:]


def _range_coding_offsets(num_noise_levels, prior_shape, dtype=tf.float32):
  """Computes the prior offsets for building range coding tables."""
  offset_indexes = tf.range(num_noise_levels, dtype=dtype)
  offset_indexes = tf.reshape(
      offset_indexes, [-1] + [1] * prior_shape.rank)
  offset = _offset_indexes_to_offset(
      offset_indexes, num_noise_levels, dtype)
  return offset


class UniversalBatchedEntropyModel(continuous_base.ContinuousEntropyModelBase):
  """Batched entropy model model which implements Universal Quantization.

  In contrast to the base class, which uses rounding for quantization, here
  "quantization" is performed additive uniform noise, which is implemented with
  Universal Quantization.

  This is described in Sec. 3.2. in the paper
  > "Universally Quantized Neural Compression"<br />
  > Eirikur Agustsson & Lucas Theis<br />
  > https://arxiv.org/abs/2006.09952
  """

  def __init__(self,
               prior,
               coding_rank,
               compression=False,
               laplace_tail_mass=0.0,
               expected_grads=False,
               tail_mass=2**-8,
               range_coder_precision=12,
               num_noise_levels=15,
               stateless=False):
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
        `bits()` method sums over each coding unit.
      compression: Boolean. If set to `True`, the range coding tables used by
        `compress()` and `decompress()` will be built on instantiation. This
        assumes eager mode (throws an error if in graph mode or inside a
        `tf.function` call). If set to `False`, these two methods will not be
        accessible.
      laplace_tail_mass: Float. If positive, will augment the prior with a
        laplace mixture for training stability.
      expected_grads: If True, will use analytical expected gradients during
        backpropagation w.r.t. additive uniform noise.
      tail_mass: Float. Approximate probability mass which is encoded using an
        Elias gamma code embedded into the range coder.
      range_coder_precision: Integer. Precision passed to the range coding op.
      num_noise_levels: Integer. The number of levels used to quantize the
        uniform noise.
      stateless: Boolean. If `True`, creates range coding tables as `Tensor`s
        rather than `Variable`s. This makes the entropy model stateless and
        allows it to be constructed within a `tf.function` body. If
        `compression=False`, then `stateless=True` is implied and the provided
        value is ignored.
    """
    if prior.event_shape.rank:
      raise ValueError("`prior` must be a (batch of) scalar distribution(s).")

    super().__init__(
        coding_rank=coding_rank,
        compression=compression,
        stateless=stateless,
        expected_grads=expected_grads,
        tail_mass=tail_mass,
        dtype=prior.dtype,
        laplace_tail_mass=laplace_tail_mass,
    )
    self._prior = prior
    self._num_noise_levels = num_noise_levels
    if self.coding_rank < self.prior_shape.rank:
      raise ValueError("`coding_rank` can't be smaller than `prior_shape`.")

    with self.name_scope:
      if self.compression:
        offset = _range_coding_offsets(
            self._num_noise_levels, self.prior_shape, self.dtype)
        cdf, cdf_offset = self._build_tables(
            self.prior, range_coder_precision, offset=offset)
        self._init_compression(cdf, cdf_offset, None)

  @property
  def prior_shape(self):
    """Batch shape of `prior` (dimensions which are not assumed i.i.d.)."""
    return tf.TensorShape(self.prior.batch_shape)

  @property
  def prior_shape_tensor(self):
    """Batch shape of `prior` as a `Tensor`."""
    return tf.constant(self.prior_shape.as_list(), dtype=tf.int32)

  def _compute_indexes_and_offset(self, broadcast_shape):
    """Returns the indexes for range coding and the quantization offset."""
    prior_size = int(self.prior_shape.num_elements())
    # Create index for each dimension in prior_shape.
    indexes = tf.range(prior_size, dtype=tf.int32)
    indexes = tf.broadcast_to(
        indexes, tf.concat((broadcast_shape, tf.shape(indexes)), axis=0))
    indexes = indexes[..., None]

    # Add in offset indexes.
    indexes = _add_offset_indexes(indexes, self._num_noise_levels)
    offset_indexes = indexes[..., 0]
    offset = _offset_indexes_to_offset(offset_indexes, self._num_noise_levels,
                                       self.dtype)

    # Flatten prior + offset indexes.
    index_ranges = [self._num_noise_levels, prior_size]
    strides = tf.math.cumprod(index_ranges, exclusive=True, reverse=True)
    indexes = tf.linalg.tensordot(indexes, strides, [[-1], [0]])
    # Now bring to full shape.
    full_shape = tf.concat([broadcast_shape, self.prior_shape_tensor], 0)
    indexes = tf.reshape(indexes, full_shape)
    offset = tf.reshape(offset, full_shape)
    return indexes, offset

  @tf.Module.with_name_scope
  def __call__(self, bottleneck, training=True):
    """Perturbs a tensor with additive uniform noise and estimates bitcost.

    Args:
      bottleneck: `tf.Tensor` containing a non-perturbed bottleneck. Must have
        at least `self.coding_rank` dimensions.
      training: Boolean. If `False`, computes the bitcost using discretized
       uniform noise. If `True`, estimates the differential entropy with uniform
       noise.

    Returns:
      A tuple
      (bottleneck_perturbed, bits)
      where `bottleneck_perturbed` is `bottleneck` perturbed with nosie
      and `bits` is the bitcost of transmitting such a sample having the same
      shape as `bottleneck` without the `self.coding_rank` innermost dimensions.
    """
    log_prob_fn = functools.partial(self._log_prob, self.prior)
    if training:
      log_probs, bottleneck_perturbed = math_ops.perturb_and_apply(
          log_prob_fn, bottleneck, expected_grads=self._expected_grads)
    else:
      # Here we compute `H(round(bottleneck - noise) | noise )`.
      input_shape = tf.shape(bottleneck)
      input_rank = tf.shape(input_shape)[0]
      _, coding_shape = tf.split(
          input_shape, [input_rank - self.coding_rank, self.coding_rank])
      broadcast_shape = coding_shape[:self.coding_rank - self.prior_shape.rank]
      _, offset = self._compute_indexes_and_offset(broadcast_shape)
      symbols = tf.round(bottleneck - offset)
      bottleneck_perturbed = symbols + offset
      log_probs = log_prob_fn(bottleneck_perturbed)

    axes = tuple(range(-self.coding_rank, 0))
    bits = tf.reduce_sum(log_probs, axis=axes) / (
        -tf.math.log(tf.constant(2., dtype=log_probs.dtype)))
    return bottleneck_perturbed, bits

  @tf.Module.with_name_scope
  def compress(self, bottleneck):
    """Compresses a floating-point tensor.

    Compresses the tensor to bit strings. `bottleneck` is first quantized
    as in `quantize()`, and then compressed using the probability tables in
    `self.cdf` derived from `self.prior`. The quantized tensor can later be
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
    input_rank = tf.shape(input_shape)[0]
    batch_shape, coding_shape = tf.split(
        input_shape, [input_rank - self.coding_rank, self.coding_rank])
    broadcast_shape = coding_shape[
        :self.coding_rank - len(self.prior_shape)]
    indexes, offset = self._compute_indexes_and_offset(broadcast_shape)
    bottleneck -= offset
    symbols = tf.cast(tf.round(bottleneck), tf.int32)
    symbols -= tf.gather(self.cdf_offset, indexes)
    handle = gen_ops.create_range_encoder(batch_shape, self.cdf)
    encode_indexes = tf.broadcast_to(indexes, tf.shape(symbols))
    handle = gen_ops.entropy_encode_index(handle, encode_indexes, symbols)
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
    decode_shape = tf.concat([broadcast_shape, self.prior_shape_tensor], 0)
    output_shape = tf.concat([tf.shape(strings), decode_shape], 0)
    indexes, offset = self._compute_indexes_and_offset(broadcast_shape)
    handle = gen_ops.create_range_decoder(strings, self.cdf)
    decode_indexes = tf.broadcast_to(indexes, output_shape)
    handle, symbols = gen_ops.entropy_decode_index(
        handle, decode_indexes, decode_shape, self.cdf_offset.dtype)
    sanity = gen_ops.entropy_decode_finalize(handle)
    tf.debugging.assert_equal(sanity, True, message="Sanity check failed.")
    symbols += tf.gather(self.cdf_offset, indexes)
    outputs = tf.cast(symbols, self.dtype)
    return outputs + offset

  def get_config(self):
    # TODO(relational): Implement this when we need serialization.
    raise NotImplementedError()


class UniversalIndexedEntropyModel(continuous_base.ContinuousEntropyModelBase):
  """Indexed entropy model model which implements Universal Quantization.

  In contrast to the base class, which uses rounding for quantization, here
  "quantization" is performed additive uniform noise, which is implemented with
  Universal Quantization.

  This is described in Sec. 3.2. in the paper
  > "Universally Quantized Neural Compression"<br />
  > Eirikur Agustsson & Lucas Theis<br />
  > https://arxiv.org/abs/2006.09952
  """

  def __init__(self,
               prior_fn,
               index_ranges,
               parameter_fns,
               coding_rank,
               compression=False,
               dtype=tf.float32,
               laplace_tail_mass=0.0,
               expected_grads=False,
               tail_mass=2**-8,
               range_coder_precision=12,
               stateless=False,
               num_noise_levels=15):
    """Initializes the instance.

    Args:
      prior_fn: A callable returning a `tfp.distributions.Distribution` object,
        typically a `Distribution` class or factory function. This is a density
        model fitting the marginal distribution of the bottleneck data with
        additive uniform noise, which is shared a priori between the sender and
        the receiver. For best results, the distributions should be flexible
        enough to have a unit-width uniform distribution as a special case,
        since this is the marginal distribution for bottleneck dimensions that
        are constant. The callable will receive keyword arguments as determined
        by `parameter_fns`.
      index_ranges: Iterable of integers. Compared to `bottleneck`, `indexes`
        in `__call__()` must have an additional trailing dimension, and the
        values of the `k`th channel must be in the range `[0, index_ranges[k])`.
      parameter_fns: Dict of strings to callables. Functions mapping `indexes`
        to each distribution parameter. For each item, `indexes` is passed to
        the callable, and the string key and return value make up one keyword
        argument to `prior_fn`.
      coding_rank: Integer. Number of innermost dimensions considered a coding
        unit. Each coding unit is compressed to its own bit string, and the
        `bits()` method sums over each coding unit.
      compression: Boolean. If set to `True`, the range coding tables used by
        `compress()` and `decompress()` will be built on instantiation. This
        assumes eager mode (throws an error if in graph mode or inside a
        `tf.function` call). If set to `False`, these two methods will not be
        accessible.
      dtype: `tf.dtypes.DType`. The data type of all floating-point computations
        carried out in this class.
      laplace_tail_mass: Float. If positive, will augment the prior with a
        laplace mixture for training stability. (experimental)
      expected_grads: If True, will use analytical expected gradients during
        backpropagation w.r.t. additive uniform noise.
      tail_mass: Float. Approximate probability mass which is encoded using an
        Elias gamma code embedded into the range coder.
      range_coder_precision: Integer. Precision passed to the range coding op.
      stateless: Boolean. If True, creates range coding tables as `Tensor`s
        rather than `Variable`s.
      num_noise_levels: Integer. The number of levels used to quantize the
        uniform noise.
    """
    if coding_rank <= 0:
      raise ValueError("`coding_rank` must be larger than 0.")
    if not callable(prior_fn):
      raise TypeError("`prior_fn` must be a class or factory function.")
    for name, fn in parameter_fns.items():
      if not isinstance(name, str):
        raise TypeError("`parameter_fns` must have string keys.")
      if not callable(fn):
        raise TypeError(f"`parameter_fns['{name}']` must be callable.")

    super().__init__(
        coding_rank=coding_rank,
        compression=compression,
        stateless=stateless,
        expected_grads=expected_grads,
        tail_mass=tail_mass,
        dtype=dtype,
        laplace_tail_mass=laplace_tail_mass,
    )
    # Add extra indexes for noise levels.
    self._index_ranges = tuple(
        [num_noise_levels] + [int(r) for r in index_ranges])
    if not self.index_ranges:
      raise ValueError("`index_ranges` must have at least one element.")
    self._prior_fn = prior_fn
    self._parameter_fns = dict(parameter_fns)
    self._num_noise_levels = num_noise_levels

    with self.name_scope:
      if self.compression:
        index_ranges = self.index_ranges_without_offsets
        indexes = [tf.range(r, dtype=self.dtype) for r in index_ranges]
        indexes = tf.meshgrid(*indexes, indexing="ij")
        indexes = tf.stack(indexes, axis=-1)
        self._prior = self._make_prior(indexes)
        offset = _range_coding_offsets(
            self._num_noise_levels, self.prior.batch_shape, self.dtype)
        cdf, cdf_offset = self._build_tables(
            self.prior, range_coder_precision, offset=offset)
        self._init_compression(cdf, cdf_offset, None)

  @property
  def index_ranges(self):
    """Upper bound(s) on values allowed in `indexes` tensor."""
    return self._index_ranges

  @property
  def parameter_fns(self):
    """Functions mapping `indexes` to each distribution parameter."""
    return self._parameter_fns

  @property
  def prior_fn(self):
    """Class or factory function returning a `Distribution` object."""
    return self._prior_fn

  @property
  def index_ranges_without_offsets(self):
    """Upper bound(s) on values allowed in `indexes` , excluding offsets."""
    return _index_ranges_without_offsets(self.index_ranges)

  def _make_prior(self, indexes):
    indexes = tf.cast(indexes, self.dtype)
    parameters = {k: f(indexes) for k, f in self.parameter_fns.items()}
    return self.prior_fn(**parameters)

  def _flatten_indexes(self, indexes):
    indexes = tf.cast(indexes, tf.int32)
    strides = tf.math.cumprod(self.index_ranges, exclusive=True, reverse=True)
    return tf.linalg.tensordot(indexes, strides, [[-1], [0]])

  def _normalize_indexes(self, indexes):
    """See base class."""
    num_indexes = indexes.shape[-1]  # Last dim of `indexes` should be static.
    if num_indexes == len(self.index_ranges):
      # Indexes have offsets.
      index_ranges = self.index_ranges
    else:
      # Indexes do not have offsets.
      index_ranges = self.index_ranges_without_offsets
      assert num_indexes == len(index_ranges)
    indexes = math_ops.lower_bound(indexes, 0)
    axes = [1] * indexes.shape.rank
    axes[-1] = len(index_ranges)
    bounds = tf.reshape([s - 1 for s in index_ranges], axes)
    return math_ops.upper_bound(indexes, tf.cast(bounds, indexes.dtype))

  def _offset_from_indexes(self, indexes_with_offsets):
    """Computes the offset for universal quantization."""
    offset_indexes = indexes_with_offsets[..., 0]
    offset = _offset_indexes_to_offset(
        offset_indexes, self._num_noise_levels, dtype=self.dtype)
    return offset

  @tf.Module.with_name_scope
  def __call__(self, bottleneck, indexes, training=True):
    """Perturbs a tensor with additive uniform noise and estimates bitcost.

    Args:
      bottleneck: `tf.Tensor` containing a non-perturbed bottleneck. Must have
        at least `self.coding_rank` dimensions.
      indexes: `tf.Tensor` specifying the scalar distribution for each element
        in `bottleneck`. See class docstring for examples.
      training: Boolean. If `False`, computes the bitcost using discretized
       uniform noise. If `True`, estimates the differential entropy with uniform
       noise.

    Returns:
      A tuple
      (bottleneck_perturbed, bits)
      where `bottleneck_perturbed` is `bottleneck` perturbed with nosie
      and `bits` is the bitcost of transmitting such a sample having the same
      shape as `bottleneck` without the `self.coding_rank` innermost dimensions.
    """
    indexes = self._normalize_indexes(indexes)
    if training:
      # Here we compute `h(bottleneck + noise)`.
      def log_prob_fn(bottleneck_perturbed, indexes):
        # When using expected_grads=True, we will use a tf.custom_gradient on
        # this function. In this case, all non-Variable tensors that determine
        # the result of this function need to be declared explicitly, i.e we
        # need `indexes` to be a declared argument and `prior` instantiated
        # here. If we would instantiate it outside this function declaration and
        # reference here via a closure, we would get a `None` gradient for
        # `indexes`.
        prior = self._make_prior(indexes)
        return self._log_prob(prior, bottleneck_perturbed)

      log_probs, bottleneck_perturbed = math_ops.perturb_and_apply(
          log_prob_fn, bottleneck, indexes, expected_grads=self._expected_grads)
    else:
      prior = self._make_prior(indexes)
      # Here we compute `H(round(bottleneck - noise) | noise )`.
      offset = _offset_indexes_to_offset(
          _add_offset_indexes(indexes, self._num_noise_levels)[..., 0],
          self._num_noise_levels, self.dtype)
      symbols = tf.round(bottleneck - offset)
      bottleneck_perturbed = symbols + offset
      log_probs = self._log_prob(prior, bottleneck_perturbed)

    axes = tuple(range(-self.coding_rank, 0))
    bits = tf.reduce_sum(log_probs, axis=axes) / (
        -tf.math.log(tf.constant(2., dtype=log_probs.dtype)))
    return bottleneck_perturbed, bits

  @tf.Module.with_name_scope
  def compress(self, bottleneck, indexes):
    """Compresses a floating-point tensor.

    Compresses the tensor to bit strings. `bottleneck` is first quantized
    as in `quantize()`, and then compressed using the probability tables derived
    from `indexes`. The quantized tensor can later be recovered by calling
    `decompress()`.

    The innermost `self.coding_rank` dimensions are treated as one coding unit,
    i.e. are compressed into one string each. Any additional dimensions to the
    left are treated as batch dimensions.

    Args:
      bottleneck: `tf.Tensor` containing the data to be compressed.
      indexes: `tf.Tensor` specifying the scalar distribution for each element
        in `bottleneck`. See class docstring for examples.

    Returns:
      A `tf.Tensor` having the same shape as `bottleneck` without the
      `self.coding_rank` innermost dimensions, containing a string for each
      coding unit.
    """
    indexes = _add_offset_indexes(indexes, self._num_noise_levels)
    indexes = self._normalize_indexes(indexes)
    flat_indexes = self._flatten_indexes(indexes)
    symbols_shape = tf.shape(flat_indexes)
    batch_shape = symbols_shape[:-self.coding_rank]
    offset = self._offset_from_indexes(indexes)
    symbols = tf.cast(tf.round(bottleneck - offset), tf.int32)
    symbols -= tf.gather(self.cdf_offset, flat_indexes)
    handle = gen_ops.create_range_encoder(batch_shape, self.cdf)
    handle = gen_ops.entropy_encode_index(handle, flat_indexes, symbols)
    return gen_ops.entropy_encode_finalize(handle)

  @tf.Module.with_name_scope
  def decompress(self, strings, indexes):
    """Decompresses a tensor.

    Reconstructs the quantized tensor from bit strings produced by `compress()`.

    Args:
      strings: `tf.Tensor` containing the compressed bit strings.
      indexes: `tf.Tensor` specifying the scalar distribution for each output
        element. See class docstring for examples.

    Returns:
      A `tf.Tensor` of the same shape as `indexes` (without the optional channel
      dimension).
    """
    indexes = _add_offset_indexes(indexes, self._num_noise_levels)
    indexes = self._normalize_indexes(indexes)
    flat_indexes = self._flatten_indexes(indexes)
    symbols_shape = tf.shape(flat_indexes)
    decode_shape = symbols_shape[-self.coding_rank:]
    handle = gen_ops.create_range_decoder(strings, self.cdf)
    handle, symbols = gen_ops.entropy_decode_index(
        handle, flat_indexes, decode_shape, self.cdf_offset.dtype)
    sanity = gen_ops.entropy_decode_finalize(handle)
    tf.debugging.assert_equal(sanity, True, message="Sanity check failed.")
    symbols += tf.gather(self.cdf_offset, flat_indexes)
    offset = self._offset_from_indexes(indexes)
    return tf.cast(symbols, self.dtype) + offset

  def get_config(self):
    # TODO(relational): Implement this when we need serialization.
    raise NotImplementedError()
