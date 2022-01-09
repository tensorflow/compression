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
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_compression.python.distributions import helpers
from tensorflow_compression.python.ops import gen_ops


__all__ = [
    "ContinuousEntropyModelBase",
]


class ContinuousEntropyModelBase(tf.Module, metaclass=abc.ABCMeta):
  """Base class for continuous entropy models.

  The basic functionality of this class is to pre-compute integer probability
  tables based on the provided `tfp.distributions.Distribution` object, which
  can then be used reliably across different platforms by the range coder and
  decoder.
  """

  @abc.abstractmethod
  def __init__(self,
               coding_rank=None,
               compression=False,
               stateless=False,
               expected_grads=False,
               tail_mass=2**-8,
               range_coder_precision=12,
               dtype=None,
               laplace_tail_mass=0):
    """Initializes the instance.

    Args:
      coding_rank: Integer. Number of innermost dimensions considered a coding
        unit. Each coding unit is compressed to its own bit string, and the
        `__call__()` method sums over each coding unit.
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
      dtype: `tf.dtypes.DType`. Data type of this entropy model (i.e. dtype of
        prior, decompressed values).
      laplace_tail_mass: Float. If positive, will augment the prior with a
        Laplace mixture for training stability. (experimental)
    """
    super().__init__()
    self._prior = None  # This will be set by subclasses, if appropriate.
    self._coding_rank = int(coding_rank)
    self._compression = bool(compression)
    self._stateless = bool(stateless)
    self._expected_grads = bool(expected_grads)
    self._tail_mass = float(tail_mass)
    self._range_coder_precision = int(range_coder_precision)
    self._dtype = tf.as_dtype(dtype)
    self._laplace_tail_mass = float(laplace_tail_mass)
    with self.name_scope:
      self._laplace_prior = (tfp.distributions.Laplace(loc=0.0, scale=1.0)
                             if laplace_tail_mass else None)

  def _check_compression(self):
    if not self.compression:
      raise RuntimeError(
          "For range coding, the entropy model must be instantiated with "
          "`compression=True`.")

  @property
  def prior(self):
    """Prior distribution, used for deriving range coding tables."""
    if self._prior is None:
      raise RuntimeError(
          "This entropy model doesn't hold a reference to its prior "
          "distribution. This can happen depending on how it is instantiated, "
          "(e.g., if it is unserialized).")
    return self._prior

  @prior.deleter
  def prior(self):
    self._prior = None

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
  def expected_grads(self):
    """Whether to use analytical expected gradients during backpropagation."""
    return self._expected_grads

  @property
  def laplace_tail_mass(self):
    """Whether to augment the prior with a Laplace mixture."""
    return self._laplace_tail_mass

  @property
  def coding_rank(self):
    """Number of innermost dimensions considered a coding unit."""
    return self._coding_rank

  @property
  def compression(self):
    """Whether this entropy model is prepared for compression."""
    return self._compression

  @property
  def stateless(self):
    """Whether range coding tables are created as `Tensor`s or `Variable`s."""
    return self._stateless

  @property
  def tail_mass(self):
    """Approximate probability mass which is range encoded with overflow."""
    return self._tail_mass

  @property
  def range_coder_precision(self):
    """Precision passed to range coding op."""
    return self._range_coder_precision

  def _init_compression(self, cdf, cdf_offset, cdf_length, cdf_shape):
    """Sets up this entropy model for using the range coder.

    This is done by storing `cdf`, `cdf_offset`, and `cdf_length` in
    `tf.Variable`s (`stateless=False`) or `tf.Tensor`s (`stateless=True`) as
    attributes of this object, or creating the variables as placeholders if
    `cdf_shape` is provided.

    The reason for pre-computing the tables is that they must not be
    re-generated independently on the sending and receiving side, since small
    numerical discrepancies between both sides can occur in this process. If the
    tables differ slightly, this in turn would very likely cause catastrophic
    error propagation during range decoding. For a more in-depth discussion of
    this, see:

    > "Integer Networks for Data Compression with Latent-Variable Models"<br />
    > J. Ballé, N. Johnston, D. Minnen<br />
    > https://openreview.net/forum?id=S1zz2i0cY7

    Args:
      cdf: CDF table for range coder.
      cdf_offset: CDF offset table for range coder.
      cdf_length: CDF length table for range coder.
      cdf_shape: Iterable of 2 integers, the shape of `cdf`. Mutually exclusive
        with the other three arguments. If provided, creates placeholder values
        for them.
    """
    if not ((cdf is None) == (cdf_offset is None) == (cdf_length is None) ==
            (cdf_shape is not None)):
      raise ValueError(
          "Either all of `cdf`, `cdf_offset`, and `cdf_length`; or `cdf_shape` "
          "must be provided.")
    if cdf_shape is not None:
      if self.stateless:
        raise ValueError("With `stateless=True`, can't provide `cdf_shape`.")
      cdf_shape = tuple(map(int, cdf_shape))
      if len(cdf_shape) != 2:
        raise ValueError("`cdf_shape` must consist of 2 integers.")
      zeros = tf.zeros(cdf_shape, dtype=tf.int32)
      cdf = zeros
      cdf_offset = zeros[:, 0]
      cdf_length = zeros[:, 0]
    if self.stateless:
      self._cdf = tf.convert_to_tensor(cdf, dtype=tf.int32, name="cdf")
      self._cdf_offset = tf.convert_to_tensor(
          cdf_offset, dtype=tf.int32, name="cdf_offset")
      self._cdf_length = tf.convert_to_tensor(
          cdf_length, dtype=tf.int32, name="cdf_length")
    else:
      self._cdf = tf.Variable(
          cdf, dtype=tf.int32, trainable=False, name="cdf")
      self._cdf_offset = tf.Variable(
          cdf_offset, dtype=tf.int32, trainable=False, name="cdf_offset")
      self._cdf_length = tf.Variable(
          cdf_length, dtype=tf.int32, trainable=False, name="cdf_length")

  def _build_tables(self, prior, offset=None, context_shape=None):
    """Computes integer-valued probability tables used by the range coder.

    These tables must not be re-generated independently on the sending and
    receiving side, since small numerical discrepancies between both sides can
    occur in this process. If the tables differ slightly, this in turn would
    very likely cause catastrophic error propagation during range decoding. For
    a more in-depth discussion of this, see:

    > "Integer Networks for Data Compression with Latent-Variable Models"<br />
    > J. Ballé, N. Johnston, D. Minnen<br />
    > https://openreview.net/forum?id=S1zz2i0cY7

    Args:
      prior: The `tfp.distributions.Distribution` object (see initializer).
      offset: Quantization offsets to use for sampling prior probabilities.
        Defaults to 0.
      context_shape: Shape of innermost dimensions to evaluate the prior on.
        Defaults to and must include `prior.batch_shape`.

    Returns:
      CDF table, CDF offsets, CDF lengths.
    """
    if offset is None:
      offset = 0.
    if context_shape is None:
      context_shape = tf.TensorShape(prior.batch_shape)
    # Subclasses should have already caught this, but better be safe.
    assert not prior.event_shape.rank

    lower_tail = helpers.lower_tail(prior, self.tail_mass)
    upper_tail = helpers.upper_tail(prior, self.tail_mass)
    # Integers such that:
    # minima + offset < lower_tail
    # maxima + offset > upper_tail
    minima = tf.cast(tf.math.floor(lower_tail - offset), tf.int32)
    maxima = tf.cast(tf.math.ceil(upper_tail - offset), tf.int32)

    # PMF starting positions and lengths.
    pmf_start = tf.cast(minima, self.dtype) + offset
    pmf_length = maxima - minima + 1

    # Sample the densities in the computed ranges, possibly computing more
    # samples than necessary at the upper end.
    max_length = tf.math.reduce_max(pmf_length)
    if tf.executing_eagerly() and max_length > 2048:
      logging.warning(
          "Very wide PMF with %d elements may lead to out of memory issues. "
          "Consider priors with smaller variance, or increasing `tail_mass` "
          "parameter.", int(max_length))
    samples = tf.range(tf.cast(max_length, self.dtype), dtype=self.dtype)
    samples = tf.reshape(samples, [-1] + context_shape.rank * [1])
    samples += pmf_start
    pmf = prior.prob(samples)

    # Collapse batch dimensions of distribution.
    pmf = tf.reshape(pmf, [max_length, -1])
    pmf = tf.transpose(pmf)

    context_shape = tf.constant(context_shape.as_list(), dtype=tf.int32)
    pmf_length = tf.broadcast_to(pmf_length, context_shape)
    pmf_length = tf.reshape(pmf_length, [-1])
    cdf_length = pmf_length + 2
    cdf_offset = tf.broadcast_to(minima, context_shape)
    cdf_offset = tf.reshape(cdf_offset, [-1])

    # Prevent tensors from bouncing back and forth between host and GPU.
    with tf.device("/cpu:0"):
      def loop_body(args):
        prob, length = args
        prob = prob[:length]
        overflow = tf.math.maximum(1 - tf.reduce_sum(prob, keepdims=True), 0.)
        prob = tf.concat([prob, overflow], axis=0)
        cdf = gen_ops.pmf_to_quantized_cdf(
            tf.cast(prob, tf.float32), precision=self.range_coder_precision)
        return tf.pad(
            cdf, [[0, max_length - length]], mode="CONSTANT", constant_values=0)

      # TODO(jonycgn,ssjhv): Consider switching to Python control flow.
      cdf = tf.map_fn(
          loop_body, (pmf, pmf_length), dtype=tf.int32, name="pmf_to_cdf")

    return cdf, cdf_offset, cdf_length

  def _log_prob(self, prior, bottleneck_perturbed):
    """Evaluates prior.log_prob(bottleneck + noise)."""
    if self.laplace_tail_mass:
      laplace_prior = self._laplace_prior
      probs = prior.prob(bottleneck_perturbed)
      probs = ((1 - self.laplace_tail_mass) * probs +
               self.laplace_tail_mass *
               laplace_prior.prob(bottleneck_perturbed))
      probs_too_small = probs < 1e-10
      probs_bounded = tf.maximum(probs, 1e-10)
      return tf.where(
          probs_too_small,
          tf.math.log(self.laplace_tail_mass) +
          laplace_prior.log_prob(bottleneck_perturbed),
          tf.math.log(probs_bounded))
    else:
      return prior.log_prob(bottleneck_perturbed)

  @abc.abstractmethod
  def get_config(self):
    """Returns the configuration of the entropy model.

    Returns:
      A JSON-serializable Python dict.

    Raises:
      RuntimeError: on attempting to call this method on an entropy model
        with `compression=False` or with `stateless=True`.
    """
    if self.stateless or not self.compression:
      raise RuntimeError(
          "Serializing entropy models with `compression=False` or "
          "`stateless=True` is not supported.")
    return dict(
        coding_rank=self.coding_rank,
        compression=True,
        stateless=False,
        expected_grads=self.expected_grads,
        tail_mass=self.tail_mass,
        range_coder_precision=self.range_coder_precision,
        cdf_shape=tuple(map(int, self.cdf.shape)),
        dtype=self.dtype.name,
        laplace_tail_mass=self.laplace_tail_mass,
    )

  def get_weights(self):
    return tf.keras.backend.batch_get_value(self.variables)

  def set_weights(self, weights):
    if len(weights) != len(self.variables):
      raise ValueError(
          "`set_weights` expects a list of {} arrays, received {}."
          "".format(len(self.variables), len(weights)))
    tf.keras.backend.batch_set_value(zip(self.variables, weights))
