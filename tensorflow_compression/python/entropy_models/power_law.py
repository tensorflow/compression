# Copyright 2022 Google LLC. All Rights Reserved.
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
"""An entropy model for the run-length gamma code."""

import tensorflow as tf
from tensorflow_compression.python.ops import gen_ops
from tensorflow_compression.python.ops import round_ops


__all__ = [
    "PowerLawEntropyModel",
]


class PowerLawEntropyModel(tf.Module):
  """Entropy model for power-law distributed random variables.

  This entropy model handles quantization and compression of a bottleneck tensor
  and implements a penalty that encourages compressibility under the Elias gamma
  code.

  The gamma code has code lengths `1 + 2 floor(log_2(x))`, for `x` a positive
  integer, and is close to optimal if `x` is distributed according to a power
  law. Being a universal code, it also guarantees that in the worst case, the
  expected code length is no more than 3 times the entropy of the empirical
  distribution of `x`, as long as probability decreases with increasing `x`. For
  details on the gamma code, see:

  > "Universal Codeword Sets and Representations of the Integers"<br />
  > P. Elias<br />
  > https://doi.org/10.1109/TIT.1975.1055349

  Given a signed integer, `run_length_gamma_encode` encodes zeros using a
  run-length code, the sign using a uniform bit, and applies the gamma code to
  the magnitude.

  The penalty applied by this class is given by:
  ```
  log((abs(x) + alpha) / alpha)
  ```
  This encourages `x` to follow a symmetrized power law, but only approximately
  for `alpha > 0`. Without `alpha`, the penalty would have a singularity at
  zero. Setting `alpha` to a small positive value ensures that the penalty is
  non-negative, and that its gradients are useful for optimization.
  """

  def __init__(self,
               coding_rank,
               alpha=1e-2,
               bottleneck_dtype=None):
    """Initializes the instance.

    Args:
      coding_rank: Integer. Number of innermost dimensions considered a coding
        unit. Each coding unit is compressed to its own bit string, and the
        estimated rate is summed over each coding unit in `bits()`.
      alpha: Float. Regularization parameter preventing gradient singularity
        around zero.
      bottleneck_dtype: `tf.dtypes.DType`. Data type of bottleneck tensor.
        Defaults to `tf.keras.mixed_precision.global_policy().compute_dtype`.
    """
    self._coding_rank = int(coding_rank)
    if self.coding_rank < 0:
      raise ValueError("`coding_rank` must be at least 0.")
    self._alpha = float(alpha)
    if self.alpha <= 0:
      raise ValueError("`alpha` must be greater than 0.")
    if bottleneck_dtype is None:
      bottleneck_dtype = tf.keras.mixed_precision.global_policy().compute_dtype
    if bottleneck_dtype is None:
      bottleneck_dtype = tf.keras.backend.floatx()
    self._bottleneck_dtype = tf.as_dtype(bottleneck_dtype)
    super().__init__()

  @property
  def alpha(self):
    """Alpha parameter."""
    return self._alpha

  @property
  def bottleneck_dtype(self):
    """Data type of the bottleneck tensor."""
    return self._bottleneck_dtype

  @property
  def coding_rank(self):
    """Number of innermost dimensions considered a coding unit."""
    return self._coding_rank

  @tf.Module.with_name_scope
  def __call__(self, bottleneck):
    """Perturbs a tensor with (quantization) noise and computes penalty.

    Args:
      bottleneck: `tf.Tensor` containing the data to be compressed. Must have at
        least `self.coding_rank` dimensions.

    Returns:
      A tuple `(self.quantize(bottleneck), self.penalty(bottleneck))`.
    """
    bottleneck = tf.convert_to_tensor(bottleneck, dtype=self.bottleneck_dtype)
    return self.quantize(bottleneck), self.penalty(bottleneck)

  @tf.Module.with_name_scope
  def penalty(self, bottleneck):
    """Computes penalty encouraging compressibility.

    Args:
      bottleneck: `tf.Tensor` containing the data to be compressed. Must have at
        least `self.coding_rank` dimensions.

    Returns:
      Penalty value, which has the same shape as `bottleneck` without the
      `self.coding_rank` innermost dimensions.
    """
    bottleneck = tf.convert_to_tensor(bottleneck, dtype=self.bottleneck_dtype)
    penalty = tf.math.log((abs(bottleneck) + self.alpha) / self.alpha)
    return tf.reduce_sum(penalty, axis=tuple(range(-self.coding_rank, 0)))

  @tf.Module.with_name_scope
  def quantize(self, bottleneck):
    """Quantizes a floating-point bottleneck tensor.

    The tensor is rounded to integer values. The gradient of this rounding
    operation is overridden with the identity (straight-through gradient
    estimator).

    Args:
      bottleneck: `tf.Tensor` containing the data to be quantized.

    Returns:
      A `tf.Tensor` containing the quantized values.
    """
    bottleneck = tf.convert_to_tensor(bottleneck, dtype=self.bottleneck_dtype)
    return round_ops.round_st(bottleneck)

  @tf.Module.with_name_scope
  def compress(self, bottleneck):
    """Compresses a floating-point tensor.

    Compresses the tensor to bit strings. `bottleneck` is first quantized
    as in `quantize()`, and then compressed using the run-length gamma code. The
    quantized tensor can later be recovered by calling `decompress()`.

    The innermost `self.coding_rank` dimensions are treated as one coding unit,
    i.e. are compressed into one string each. Any additional dimensions to the
    left are treated as batch dimensions.

    Args:
      bottleneck: `tf.Tensor` containing the data to be compressed. Must have at
        least `self.coding_rank` dimensions.

    Returns:
      A `tf.Tensor` having the same shape as `bottleneck` without the
      `self.coding_rank` innermost dimensions, containing a string for each
      coding unit.
    """
    bottleneck = tf.convert_to_tensor(bottleneck, dtype=self.bottleneck_dtype)

    shape = tf.shape(bottleneck)
    if self.coding_rank == 0:
      flat_shape = [-1]
      strings_shape = shape
    else:
      flat_shape = tf.concat([[-1], shape[-self.coding_rank:]], 0)
      strings_shape = shape[:-self.coding_rank]

    symbols = tf.cast(tf.round(bottleneck), tf.int32)
    symbols = tf.reshape(symbols, flat_shape)

    strings = tf.map_fn(
        gen_ops.run_length_gamma_encode, symbols,
        fn_output_signature=tf.TensorSpec((), dtype=tf.string))
    return tf.reshape(strings, strings_shape)

  @tf.Module.with_name_scope
  def decompress(self, strings, code_shape):
    """Decompresses a tensor.

    Reconstructs the quantized tensor from bit strings produced by `compress()`.

    Args:
      strings: `tf.Tensor` containing the compressed bit strings.
      code_shape: Shape of innermost dimensions of the output `tf.Tensor`.

    Returns:
      A `tf.Tensor` of shape `tf.shape(strings) + code_shape`.
    """
    strings = tf.convert_to_tensor(strings, dtype=tf.string)
    strings_shape = tf.shape(strings)
    symbols = tf.map_fn(
        lambda x: gen_ops.run_length_gamma_decode(x, code_shape),
        tf.reshape(strings, [-1]),
        fn_output_signature=tf.TensorSpec(
            [None] * self.coding_rank, dtype=tf.int32))
    symbols = tf.reshape(symbols, tf.concat([strings_shape, code_shape], 0))
    return tf.cast(symbols, self.bottleneck_dtype)
