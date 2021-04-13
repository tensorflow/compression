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
"""Indexed entropy model for continuous random variables."""

import tensorflow as tf
from tensorflow_compression.python.distributions import helpers
from tensorflow_compression.python.entropy_models import continuous_base
from tensorflow_compression.python.ops import gen_ops
from tensorflow_compression.python.ops import math_ops


__all__ = [
    "ContinuousIndexedEntropyModel",
    "LocationScaleIndexedEntropyModel",
]


class ContinuousIndexedEntropyModel(continuous_base.ContinuousEntropyModelBase):
  """Indexed entropy model for continuous random variables.

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
    using `training=False`.
  - Instantiate an entropy model with `compression=True` (and the same
    parameters as during training), and share the model between a sender and a
    receiver.
  - On the sender side, compute the bottleneck tensor and call `compress()` on
    it. The output is a compressed string representation of the tensor. Transmit
    the string to the receiver, and call `decompress()` there. The output is the
    quantized bottleneck tensor. Continue processing the tensor on the receiving
    side.

  This class assumes that all scalar elements of the encoded tensor are
  conditionally independent given some other random variable, possibly depending
  on data. All dependencies must be represented by the `indexes` tensor. For
  each bottleneck tensor element, it selects the appropriate scalar
  distribution.

  The `indexes` tensor must contain only integer values in a pre-specified range
  (but may have floating-point type for purposes of backpropagation). To make
  the distribution conditional on `n`-dimensional indexes, `index_ranges` must
  be specified as an iterable of `n` integers. `indexes` must have the same
  shape as the bottleneck tensor with an additional channel dimension of length
  `n`. The position of the channel dimension is given by `channel_axis`. The
  index values in the `k`th channel must be in the range `[0, index_ranges[k])`.
  If `index_ranges` has only one element (i.e. `n == 1`), `channel_axis` may be
  `None`. In that case, the additional channel dimension is omitted, and the
  `indexes` tensor must have the same shape as the bottleneck tensor.

  The implied distribution for the bottleneck tensor is determined as:
  ```
  prior_fn(**{k: f(indexes) for k, f in parameter_fns.items()})
  ```

  A more detailed description (and motivation) of this indexing scheme can be
  found in the following paper. Please cite the paper when using this code for
  derivative work.

  > "Integer Networks for Data Compression with Latent-Variable Models"<br />
  > J. Ballé, N. Johnston, D. Minnen<br />
  > https://openreview.net/forum?id=S1zz2i0cY7

  Examples:

  To make a parameterized zero-mean normal distribution, one could use:
  ```
  tfc.ContinuousIndexedEntropyModel(
      prior_fn=tfc.NoisyNormal,
      index_ranges=(64,),
      parameter_fns=dict(
          loc=lambda _: 0.,
          scale=lambda i: tf.exp(i / 8 - 5),
      ),
      coding_rank=1,
      channel_axis=None,
  )
  ```
  Then, each element of `indexes` in the range `[0, 64)` would indicate that the
  corresponding element in `bottleneck` is normally distributed with zero mean
  and a standard deviation between `exp(-5)` and `exp(2.875)`, inclusive.

  To make a parameterized logistic mixture distribution, one could use:
  ```
  tfc.ContinuousIndexedEntropyModel(
      prior_fn=tfc.NoisyLogisticMixture,
      index_ranges=(10, 10, 5),
      parameter_fns=dict(
          loc=lambda i: i[..., 0:2] - 5,
          scale=lambda _: 1,
          weight=lambda i: tf.nn.softmax((i[..., 2:3] - 2) * [-1, 1]),
      ),
      coding_rank=1,
      channel_axis=-1,
  )
  ```
  Then, the last dimension of `indexes` would consist of triples of elements in
  the ranges `[0, 10)`, `[0, 10)`, and `[0, 5)`, respectively. Each triple
  would indicate that the element in `bottleneck` corresponding to the other
  dimensions is distributed with a mixture of two logistic distributions, where
  the components each have one of 10 location parameters between `-5` and `+4`,
  inclusive, unit scale parameters, and one of five different mixture
  weightings.
  """

  def __init__(self,
               prior_fn,
               index_ranges,
               parameter_fns,
               coding_rank,
               channel_axis=-1,
               compression=False,
               stateless=False,
               expected_grads=False,
               tail_mass=2**-8,
               range_coder_precision=12,
               dtype=tf.float32,
               laplace_tail_mass=0):
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
      index_ranges: Iterable of integers. `indexes` must have the same shape as
        the bottleneck tensor, with an additional dimension at position
        `channel_axis`. The values of the `k`th channel must be in the range
        `[0, index_ranges[k])`.
      parameter_fns: Dict of strings to callables. Functions mapping `indexes`
        to each distribution parameter. For each item, `indexes` is passed to
        the callable, and the string key and return value make up one keyword
        argument to `prior_fn`.
      coding_rank: Integer. Number of innermost dimensions considered a coding
        unit. Each coding unit is compressed to its own bit string, and the
        bits in the `__call__` method are summed over each coding unit.
      channel_axis: Integer or `None`. Determines the position of the channel
        axis in `indexes`. Defaults to the last dimension. If set to `None`,
        the index tensor is expected to have the same shape as the bottleneck
        tensor (only allowed when `index_ranges` has length 1).
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
      tail_mass: Float. Approximate probability mass which is range encoded with
        less precision, by using a Golomb-like code.
      range_coder_precision: Integer. Precision passed to the range coding op.
      dtype: `tf.dtypes.DType`. The data type of all floating-point
        computations carried out in this class.
      laplace_tail_mass: Float. If positive, will augment the prior with a
        laplace mixture for training stability. (experimental)
    """
    if coding_rank <= 0:
      raise ValueError("coding_rank must be larger than 0.")
    if not callable(prior_fn):
      raise TypeError("prior_fn must be a class or factory function.")
    for name, fn in parameter_fns.items():
      if not isinstance(name, str):
        raise TypeError("parameter_fns must have string keys.")
      if not callable(fn):
        raise TypeError(f"parameter_fns['{name}'] must be callable.")
    self._index_ranges = tuple(int(r) for r in index_ranges)
    if not self.index_ranges:
      raise ValueError("index_ranges must have at least one element.")
    self._channel_axis = None if channel_axis is None else int(channel_axis)
    if self.channel_axis is None and len(self.index_ranges) > 1:
      raise ValueError("channel_axis can't be None for len(index_ranges) > 1.")
    self._prior_fn = prior_fn
    self._parameter_fns = dict(parameter_fns)
    super().__init__(
        prior=self._make_range_coding_prior(self.index_ranges, dtype),
        coding_rank=coding_rank,
        compression=compression,
        stateless=stateless,
        expected_grads=expected_grads,
        tail_mass=tail_mass,
        range_coder_precision=range_coder_precision,
        laplace_tail_mass=laplace_tail_mass,
    )

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
  def channel_axis(self):
    """Position of channel axis in `indexes` tensor."""
    return self._channel_axis

  def _make_prior(self, indexes, dtype=None):
    indexes = tf.cast(indexes, dtype or self.dtype)
    parameters = {k: f(indexes) for k, f in self.parameter_fns.items()}
    return self.prior_fn(**parameters)

  def _make_range_coding_prior(self, index_ranges, dtype):
    """Instantiates the range coding prior."""
    dtype = tf.as_dtype(dtype)
    if self.channel_axis is None:
      index_range, = index_ranges
      indexes = tf.range(index_range, dtype=dtype)
    else:
      indexes = [tf.range(r, dtype=dtype) for r in index_ranges]
      indexes = tf.meshgrid(*indexes, indexing="ij")
      indexes = tf.stack(indexes, axis=self.channel_axis)
    return self._make_prior(indexes, dtype=dtype)

  def _normalize_indexes(self, indexes):
    indexes = math_ops.lower_bound(indexes, 0)
    if self.channel_axis is None:
      index_range, = self.index_ranges
      bounds = index_range - 1
    else:
      axes = [1] * indexes.shape.rank
      axes[self.channel_axis] = len(self.index_ranges)
      bounds = tf.reshape([s - 1 for s in self.index_ranges], axes)
    indexes = math_ops.upper_bound(indexes, tf.cast(bounds, indexes.dtype))
    return indexes

  def _flatten_indexes(self, indexes):
    indexes = tf.cast(indexes, tf.int32)
    if self.channel_axis is None:
      return indexes
    else:
      strides = tf.math.cumprod(self.index_ranges, exclusive=True, reverse=True)
      return tf.linalg.tensordot(indexes, strides, [[self.channel_axis], [0]])

  def _offset_from_indexes(self, indexes):
    """Compute the quantization offset from the respective prior."""
    prior = self._make_prior(indexes)
    return helpers.quantization_offset(prior)

  @tf.Module.with_name_scope
  def __call__(self, bottleneck, indexes, training=True):
    """Perturbs a tensor with (quantization) noise and estimates rate.

    Args:
      bottleneck: `tf.Tensor` containing the data to be compressed.
      indexes: `tf.Tensor` specifying the scalar distribution for each element
        in `bottleneck`. See class docstring for examples.
      training: Boolean. If `False`, computes the Shannon information of
        `bottleneck` under the distribution computed by `self.prior_fn`,
        which is a non-differentiable, tight *lower* bound on the number of bits
        needed to compress `bottleneck` using `compress()`. If `True`, returns a
        somewhat looser, but differentiable *upper* bound on this quantity.

    Returns:
      A tuple (bottleneck_perturbed, bits) where `bottleneck_perturbed` is
      `bottleneck` perturbed with (quantization) noise and `bits` is the rate.
      `bits` has the same shape as `bottleneck` without the `self.coding_rank`
      innermost dimensions.
    """
    indexes = self._normalize_indexes(indexes)
    prior = self._make_prior(indexes)
    if training:
      bottleneck_perturbed = bottleneck + tf.random.uniform(
          tf.shape(bottleneck), minval=-.5, maxval=.5, dtype=bottleneck.dtype)
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
      offset = helpers.quantization_offset(prior)
      bottleneck_perturbed = self._quantize(bottleneck, offset)
      log_probs = self._log_prob_from_prior(prior, bottleneck_perturbed)
    axes = tuple(range(-self.coding_rank, 0))
    bits = tf.reduce_sum(log_probs, axis=axes) / (
        -tf.math.log(tf.constant(2, dtype=log_probs.dtype)))
    return bottleneck_perturbed, bits

  @tf.Module.with_name_scope
  def quantize(self, bottleneck, indexes):
    """Quantizes a floating-point tensor.

    To use this entropy model as an information bottleneck during training, pass
    a tensor through this function. The tensor is rounded to integer values
    modulo a quantization offset, which depends on `indexes`. For instance, for
    Gaussian distributions, the returned values are rounded to the location of
    the mode of the distributions plus or minus an integer.

    The gradient of this rounding operation is overridden with the identity
    (straight-through gradient estimator).

    Args:
      bottleneck: `tf.Tensor` containing the data to be quantized.
      indexes: `tf.Tensor` specifying the scalar distribution for each element
        in `bottleneck`. See class docstring for examples.

    Returns:
      A `tf.Tensor` containing the quantized values.
    """
    indexes = self._normalize_indexes(indexes)
    offset = self._offset_from_indexes(indexes)
    return self._quantize(bottleneck, offset)

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
    indexes = self._normalize_indexes(indexes)
    flat_indexes = self._flatten_indexes(indexes)

    symbols_shape = tf.shape(flat_indexes)
    batch_shape = symbols_shape[:-self.coding_rank]
    flat_shape = tf.concat([[-1], symbols_shape[-self.coding_rank:]], 0)

    flat_indexes = tf.reshape(flat_indexes, flat_shape)

    offset = self._offset_from_indexes(indexes)
    symbols = tf.cast(tf.round(bottleneck - offset), tf.int32)
    symbols = tf.reshape(symbols, flat_shape)

    # Prevent tensors from bouncing back and forth between host and GPU.
    with tf.device("/cpu:0"):
      cdf = self.cdf
      cdf_length = self.cdf_length
      cdf_offset = self.cdf_offset
      def loop_body(args):
        return gen_ops.unbounded_index_range_encode(
            args[0], args[1], cdf, cdf_length, cdf_offset,
            precision=self.range_coder_precision,
            overflow_width=4, debug_level=1)

      # TODO(jonycgn,ssjhv): Consider switching to Python control flow.
      strings = tf.map_fn(
          loop_body, (symbols, flat_indexes), dtype=tf.string, name="compress")

    strings = tf.reshape(strings, batch_shape)
    return strings

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
    indexes = self._normalize_indexes(indexes)
    flat_indexes = self._flatten_indexes(indexes)

    symbols_shape = tf.shape(flat_indexes)
    flat_shape = tf.concat([[-1], symbols_shape[-self.coding_rank:]], 0)

    flat_indexes = tf.reshape(flat_indexes, flat_shape)

    strings = tf.reshape(strings, [-1])

    # Prevent tensors from bouncing back and forth between host and GPU.
    with tf.device("/cpu:0"):
      cdf = self.cdf
      cdf_length = self.cdf_length
      cdf_offset = self.cdf_offset
      def loop_body(args):
        return gen_ops.unbounded_index_range_decode(
            args[0], args[1], cdf, cdf_length, cdf_offset,
            precision=self.range_coder_precision,
            overflow_width=4, debug_level=1)

      # TODO(jonycgn,ssjhv): Consider switching to Python control flow.
      symbols = tf.map_fn(
          loop_body, (strings, flat_indexes), dtype=tf.int32, name="decompress")

    symbols = tf.reshape(symbols, symbols_shape)
    offset = self._offset_from_indexes(indexes)
    return tf.cast(symbols, self.dtype) + offset

  def get_config(self):
    """Returns the configuration of the entropy model."""
    raise NotImplementedError(
        "Serializing indexed entropy models is not yet implemented.")

  @classmethod
  def from_config(cls, config):
    """Instantiates an entropy model from a configuration dictionary."""
    raise NotImplementedError(
        "Serializing indexed entropy models is not yet implemented.")


class LocationScaleIndexedEntropyModel(ContinuousIndexedEntropyModel):
  """Indexed entropy model for location-scale family of random variables.

  This class is a common special case of `ContinuousIndexedEntropyModel`. The
  specified distribution is parameterized with `num_scales` values of scale
  parameters. An element-wise location parameter is handled by shifting the
  distributions to zero. Note: this only works for shift-invariant
  distributions, where the `loc` parameter really denotes a translation (i.e.,
  not for the log-normal distribution).
  """

  def __init__(self,
               prior_fn,
               num_scales,
               scale_fn,
               coding_rank,
               compression=False,
               stateless=False,
               expected_grads=False,
               tail_mass=2**-8,
               range_coder_precision=12,
               dtype=tf.float32,
               laplace_tail_mass=0):
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
      num_scales: Integer. Values in `indexes` must be in the range
        `[0, num_scales)`.
      scale_fn: Callable. `indexes` is passed to the callable, and the return
        value is given as `scale` keyword argument to `prior_fn`.
      coding_rank: Integer. Number of innermost dimensions considered a coding
        unit. Each coding unit is compressed to its own bit string, and the
        bits in the `__call__` method are summed over each coding unit.
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
      tail_mass: Float. Approximate probability mass which is range encoded with
        less precision, by using a Golomb-like code.
      range_coder_precision: Integer. Precision passed to the range coding op.
      dtype: `tf.dtypes.DType`. The data type of all floating-point
        computations carried out in this class.
      laplace_tail_mass: Float. If positive, will augment the prior with a
        laplace mixture for training stability. (experimental)
    """
    num_scales = int(num_scales)
    super().__init__(
        prior_fn=prior_fn,
        index_ranges=(num_scales,),
        parameter_fns=dict(
            loc=lambda _: 0.,
            scale=scale_fn,
        ),
        coding_rank=coding_rank,
        channel_axis=None,
        compression=compression,
        stateless=stateless,
        expected_grads=expected_grads,
        tail_mass=tail_mass,
        range_coder_precision=range_coder_precision,
        dtype=dtype,
        laplace_tail_mass=laplace_tail_mass,
    )

  @tf.Module.with_name_scope
  def __call__(self, bottleneck, scale_indexes, loc=None, training=True):
    """Perturbs a tensor with (quantization) noise and estimates rate.

    Args:
      bottleneck: `tf.Tensor` containing the data to be compressed.
      scale_indexes: `tf.Tensor` indexing the scale parameter for each element
        in `bottleneck`. Must have the same shape as `bottleneck`.
      loc: `None` or `tf.Tensor`. If `None`, the location parameter for all
        elements is assumed to be zero. Otherwise, specifies the location
        parameter for each element in `bottleneck`. Must have the same shape as
        `bottleneck`.
      training: Boolean. If `False`, computes the Shannon information of
        `bottleneck` under the distribution computed by `self.prior_fn`,
        which is a non-differentiable, tight *lower* bound on the number of bits
        needed to compress `bottleneck` using `compress()`. If `True`, returns a
        somewhat looser, but differentiable *upper* bound on this quantity.

    Returns:
      A tuple (bottleneck_perturbed, bits) where `bottleneck_perturbed` is
      `bottleneck` perturbed with (quantization) noise and `bits` is the rate.
      `bits` has the same shape as `bottleneck` without the `self.coding_rank`
      innermost dimensions.
    """
    if loc is None:
      loc = 0.0
    bottleneck_centered = bottleneck - loc
    bottleneck_centered_perturbed, bits = super().__call__(
        bottleneck_centered, scale_indexes, training=training)
    bottleneck_perturbed = bottleneck_centered_perturbed + loc
    return bottleneck_perturbed, bits

  @tf.Module.with_name_scope
  def quantize(self, bottleneck, scale_indexes, loc=None):
    """Quantizes a floating-point tensor.

    To use this entropy model as an information bottleneck during training, pass
    a tensor through this function. The tensor is rounded to integer values
    modulo a quantization offset, which depends on `indexes`. For instance, for
    Gaussian distributions, the returned values are rounded to the location of
    the mode of the distributions plus or minus an integer.

    The gradient of this rounding operation is overridden with the identity
    (straight-through gradient estimator).

    Args:
      bottleneck: `tf.Tensor` containing the data to be quantized.
      scale_indexes: `tf.Tensor` indexing the scale parameter for each element
        in `bottleneck`. Must have the same shape as `bottleneck`.
      loc: `None` or `tf.Tensor`. If `None`, the location parameter for all
        elements is assumed to be zero. Otherwise, specifies the location
        parameter for each element in `bottleneck`. Must have the same shape as
        `bottleneck`.

    Returns:
      A `tf.Tensor` containing the quantized values.
    """
    if loc is None:
      return super().quantize(bottleneck, scale_indexes)
    else:
      return super().quantize(bottleneck - loc, scale_indexes) + loc

  @tf.Module.with_name_scope
  def compress(self, bottleneck, scale_indexes, loc=None):
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
      scale_indexes: `tf.Tensor` indexing the scale parameter for each element
        in `bottleneck`. Must have the same shape as `bottleneck`.
      loc: `None` or `tf.Tensor`. If `None`, the location parameter for all
        elements is assumed to be zero. Otherwise, specifies the location
        parameter for each element in `bottleneck`. Must have the same shape as
        `bottleneck`.

    Returns:
      A `tf.Tensor` having the same shape as `bottleneck` without the
      `self.coding_rank` innermost dimensions, containing a string for each
      coding unit.
    """
    if loc is not None:
      bottleneck -= loc
    return super().compress(bottleneck, scale_indexes)

  @tf.Module.with_name_scope
  def decompress(self, strings, scale_indexes, loc=None):
    """Decompresses a tensor.

    Reconstructs the quantized tensor from bit strings produced by `compress()`.

    Args:
      strings: `tf.Tensor` containing the compressed bit strings.
      scale_indexes: `tf.Tensor` indexing the scale parameter for each output
        element.
      loc: `None` or `tf.Tensor`. If `None`, the location parameter for all
        output elements is assumed to be zero. Otherwise, specifies the location
        parameter for each output element. Must have the same shape as
        `scale_indexes`.

    Returns:
      A `tf.Tensor` of the same shape as `scale_indexes`.
    """
    values = super().decompress(strings, scale_indexes)
    if loc is not None:
      values += loc
    return values
