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
from tensorflow_compression.python.entropy_models import continuous_batched
from tensorflow_compression.python.entropy_models import continuous_indexed
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


class UniversalBatchedEntropyModel(
    continuous_batched.ContinuousBatchedEntropyModel):
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
      tail_mass: Float. Approximate probability mass which is range encoded with
        less precision, by using a Golomb-like code.
      range_coder_precision: Integer. Precision passed to the range coding op.
      num_noise_levels: Integer. The number of levels used to quantize the
        uniform noise.
      stateless: Boolean. If `True`, creates range coding tables as `Tensor`s
        rather than `Variable`s. This makes the entropy model stateless and
        allows it to be constructed within a `tf.function` body. If
        `compression=False`, then `stateless=True` is implied and the provided
        value is ignored.

    Raises:
      RuntimeError: when attempting to instantiate an entropy model with
        `compression=True` and not in eager execution mode.
    """
    # This attribute is used in methods we override in this class which
    # are used during used during super().__init__(...), so we set it first.
    self._num_noise_levels = num_noise_levels

    super().__init__(
        prior=prior,
        coding_rank=coding_rank,
        compression=compression,
        laplace_tail_mass=laplace_tail_mass,
        expected_grads=expected_grads,
        tail_mass=tail_mass,
        range_coder_precision=range_coder_precision,
        stateless=stateless)

  @property
  def context_shape(self):
    """See base class."""
    return (self._num_noise_levels,) + self.prior_shape

  def _cache_quantization_offset(self):
    """See base class."""
    # Universal Quantization derives offsets from a pseudorandom source.
    self._quantization_offset = None

  def _offset_from_prior(self, prior):
    """See base class."""
    return _range_coding_offsets(self._num_noise_levels, self.prior_shape,
                                 self.dtype)

  def _compute_indexes_and_offset(self, broadcast_shape):
    """See base class."""
    prior_size = int(self.prior_shape.num_elements())
    # Create index for each dimension in prior_shape.
    indexes = tf.range(prior_size, dtype=tf.int32)
    indexes = tf.broadcast_to(
        indexes, tf.concat((broadcast_shape, tf.shape(indexes)), axis=0))
    # Add channel dimension.
    channel_axis = -1
    indexes = indexes[..., None]

    # Add in offset indexes.
    indexes = _add_offset_indexes(indexes, self._num_noise_levels)
    offset_indexes = indexes[..., 0]
    offset = _offset_indexes_to_offset(offset_indexes, self._num_noise_levels,
                                       self.dtype)

    # Flatten prior + offset indexes.
    index_ranges = [self._num_noise_levels, prior_size]
    strides = tf.math.cumprod(index_ranges, exclusive=True, reverse=True)
    indexes = tf.linalg.tensordot(indexes, strides, [[channel_axis], [0]])
    # Now bring to full shape.
    full_shape = tf.concat([broadcast_shape, self.prior_shape_tensor], 0)
    indexes = tf.reshape(indexes, full_shape)
    offset = tf.reshape(offset, full_shape)
    return indexes, offset

  @tf.Module.with_name_scope
  def quantize(self, bottleneck, indexes=None):
    raise NotImplementedError()

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

    log_prob_fn = functools.partial(self._log_prob_from_prior, self.prior)
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

  def get_config(self):
    # TODO(relational): Implement this when we need serialization.
    raise NotImplementedError()


class UniversalIndexedEntropyModel(
    continuous_indexed.ContinuousIndexedEntropyModel):
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
      tail_mass: Float. Approximate probability mass which is range encoded with
        less precision, by using a Golomb-like code.
      range_coder_precision: Integer. Precision passed to the range coding op.
      stateless: Boolean. If True, creates range coding tables as `Tensor`s
        rather than `Variable`s.
      num_noise_levels: Integer. The number of levels used to quantize the
        uniform noise.

    Raises:
      RuntimeError: when attempting to instantiate an entropy model with
        `compression=True` and not in eager execution mode.
    """
    # Add extra indexes for noise levels.
    index_ranges_with_offsets = tuple([num_noise_levels] +
                                      [int(r) for r in index_ranges])

    # This attribute is used in methods we override in this class which
    # are used during used during super().__init__(...), so we set it first.
    self._num_noise_levels = num_noise_levels

    # We only support channel axis at the last dimension.
    channel_axis = -1
    super().__init__(
        prior_fn=prior_fn,
        index_ranges=index_ranges_with_offsets,
        parameter_fns=parameter_fns,
        coding_rank=coding_rank,
        compression=compression,
        channel_axis=channel_axis,
        dtype=dtype,
        tail_mass=tail_mass,
        laplace_tail_mass=laplace_tail_mass,
        expected_grads=expected_grads,
        range_coder_precision=range_coder_precision,
        stateless=stateless)

  @property
  def context_shape(self):
    """See base class."""
    return (self._num_noise_levels,) + self.prior_shape

  @property
  def index_ranges_without_offsets(self):
    """Upper bound(s) on values allowed in `indexes` , excluding offsets."""
    return _index_ranges_without_offsets(self.index_ranges)

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
    axes[self.channel_axis] = len(index_ranges)
    bounds = tf.reshape([s - 1 for s in index_ranges], axes)
    return math_ops.upper_bound(indexes, tf.cast(bounds, indexes.dtype))

  def _offset_from_indexes(self, indexes_with_offsets):
    """Computes the offset for universal quantization (overrides base class)."""
    offset_indexes = indexes_with_offsets[..., 0]
    offset = _offset_indexes_to_offset(
        offset_indexes, self._num_noise_levels, dtype=self.dtype)
    return offset

  def _make_range_coding_prior(self, index_ranges, dtype):
    """Instantiates the range coding prior."""
    return super()._make_range_coding_prior(
        _index_ranges_without_offsets(index_ranges), dtype)

  def _offset_from_prior(self, prior):
    return _range_coding_offsets(self._num_noise_levels, self.prior_shape,
                                 self.dtype)

  @tf.Module.with_name_scope
  def quantize(self, bottleneck, indexes=None):
    raise NotImplementedError()

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
        return self._log_prob_from_prior(prior, bottleneck_perturbed)

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
      log_probs = self._log_prob_from_prior(prior, bottleneck_perturbed)

    axes = tuple(range(-self.coding_rank, 0))
    bits = tf.reduce_sum(log_probs, axis=axes) / (
        -tf.math.log(tf.constant(2., dtype=log_probs.dtype)))
    return bottleneck_perturbed, bits

  @tf.Module.with_name_scope
  def compress(self, bottleneck, indexes):
    """See base class."""
    indexes_with_offset = _add_offset_indexes(indexes, self._num_noise_levels)
    return super().compress(bottleneck, indexes_with_offset)

  @tf.Module.with_name_scope
  def decompress(self, strings, indexes):
    """See base class."""
    indexes_with_offset = _add_offset_indexes(indexes, self._num_noise_levels)
    return super().decompress(strings, indexes_with_offset)

  def get_config(self):
    # TODO(relational): Implement this when we need serialization.
    raise NotImplementedError()
