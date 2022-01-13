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
"""Extensions to `tfp.distributions.Distribution` to handle range coding."""

import tensorflow as tf


__all__ = [
    "estimate_tails",
    "quantization_offset",
    "lower_tail",
    "upper_tail",
]


# TODO(jonycgn): Consider wrapping in tf.function.
def estimate_tails(func, target, shape, dtype):
  """Estimates approximate tail quantiles.

  This runs a simple Adam iteration to determine tail quantiles. The
  objective is to find an `x` such that:
  ```
  func(x) == target
  ```
  For instance, if `func` is a CDF and the target is a quantile value, this
  would find the approximate location of that quantile. Note that `func` is
  assumed to be monotonic. When each tail estimate has passed the optimal value
  of `x`, the algorithm does 100 additional iterations and then stops.

  This operation is vectorized. The tensor shape of `x` is given by `shape`, and
  `target` must have a shape that is broadcastable to the output of `func(x)`.

  Args:
    func: A callable that computes cumulative distribution function, survival
      function, or similar.
    target: The desired target value.
    shape: The shape of the `tf.Tensor` representing `x`.
    dtype: The `tf.dtypes.Dtype` of the computation (and the return value).

  Returns:
    A `tf.Tensor` representing the solution (`x`).
  """
  with tf.name_scope("estimate_tails"):
    dtype = tf.as_dtype(dtype)
    shape = tf.convert_to_tensor(shape, tf.int32)
    target = tf.convert_to_tensor(target, dtype)

    def loop_cond(tails, m, v, count):
      del tails, m, v  # unused
      return tf.reduce_min(count) < 100

    def loop_body(tails, prev_m, prev_v, count):
      with tf.GradientTape(watch_accessed_variables=False) as tape:
        tape.watch(tails)
        loss = abs(func(tails) - target)
      grad = tape.gradient(loss, tails)
      m = (prev_m + grad) / 2  # Adam mean estimate.
      v = (prev_v + tf.square(grad)) / 2  # Adam variance estimate.
      tails -= .1 * m / (tf.sqrt(v) + 1e-20)
      # Start counting when the gradient flips sign. Since the function is
      # monotonic, m must have the same sign in all initial iterations, until
      # the optimal point is crossed. At that point the gradient flips sign.
      count = tf.where(
          tf.math.logical_or(count > 0, prev_m * grad < 0),
          count + 1, count)
      return tails, m, v, count

    init_tails = tf.zeros(shape, dtype=dtype)
    init_m = tf.zeros(shape, dtype=dtype)
    init_v = tf.ones(shape, dtype=dtype)
    init_count = tf.zeros(shape, dtype=tf.int32)
    return tf.while_loop(
        loop_cond, loop_body, (init_tails, init_m, init_v, init_count))[0]


def quantization_offset(distribution):
  """Computes distribution-dependent quantization offset.

  For range coding of continuous random variables, the values need to be
  quantized first. Typically, it is beneficial for compression performance to
  align the centers of the quantization bins such that one of them coincides
  with the mode of the distribution. With `offset` being the mode of the
  distribution, for instance, this can be achieved simply by computing:
  ```
  x_hat = tf.round(x - offset) + offset
  ```

  This method tries to determine the offset in a best-effort fashion, based on
  which statistics the `Distribution` implements. First, a method
  `self._quantization_offset()` is tried. If that isn't defined, it tries in
  turn: `self.mode()`, `self.quantile(.5)`, then `self.mean()`. If none of
  these are implemented, it falls back on quantizing to integer values (i.e.,
  an offset of zero).

  Note the offset is always in the range [-.5, .5] as it is assumed to be
  combined with a round quantizer.

  Args:
    distribution: A `tfp.distributions.Distribution` object.

  Returns:
    A `tf.Tensor` broadcastable to shape `self.batch_shape`, containing
    the determined quantization offsets. No gradients are allowed to flow
    through the return value.
  """
  try:
    offset = distribution._quantization_offset()  # pylint:disable=protected-access
  except (AttributeError, NotImplementedError):
    try:
      offset = distribution.mode()
    except NotImplementedError:
      try:
        offset = distribution.quantile(.5)
      except NotImplementedError:
        try:
          offset = distribution.mean()
        except NotImplementedError:
          offset = tf.constant(0, dtype=distribution.dtype)
  return tf.stop_gradient(offset - tf.round(offset))


def lower_tail(distribution, tail_mass):
  """Approximates lower tail quantile for range coding.

  For range coding of random variables, the distribution tails need special
  handling, because range coding can only handle alphabets with a finite
  number of symbols. This method returns a cut-off location for the lower
  tail, such that approximately `tail_mass` probability mass is contained in
  the tails (together). The tails are then handled by using the 'overflow'
  functionality of the range coder implementation (using an Elias gamma code).

  Args:
    distribution: A `tfp.distributions.Distribution` object.
    tail_mass: Float between 0 and 1. Desired probability mass for the tails.

  Returns:
    A `tf.Tensor` broadcastable to shape `self.batch_shape` containing the
    approximate lower tail quantiles for each scalar distribution.
  """
  try:
    tail = distribution._lower_tail(tail_mass)  # pylint:disable=protected-access
  except (AttributeError, NotImplementedError):
    try:
      tail = distribution.quantile(tail_mass / 2)
    except NotImplementedError:
      try:
        target = tf.math.log(tf.cast(tail_mass / 2, distribution.dtype))
        tail = estimate_tails(
            distribution.log_cdf, target, distribution.batch_shape_tensor(),
            distribution.dtype)
      except NotImplementedError:
        raise NotImplementedError(
            "`distribution` must implement `_lower_tail()`, `quantile()`, or "
            "`log_cdf()` so that lower tail can be located.")
  return tf.stop_gradient(tail)


def upper_tail(distribution, tail_mass):
  """Approximates upper tail quantile for range coding.

  For range coding of random variables, the distribution tails need special
  handling, because range coding can only handle alphabets with a finite
  number of symbols. This method returns a cut-off location for the upper
  tail, such that approximately `tail_mass` probability mass is contained in
  the tails (together). The tails are then handled by using the 'overflow'
  functionality of the range coder implementation (using an Elias gamma code).

  Args:
    distribution: A `tfp.distributions.Distribution` object.
    tail_mass: Float between 0 and 1. Desired probability mass for the tails.

  Returns:
    A `tf.Tensor` broadcastable to shape `self.batch_shape` containing the
    approximate upper tail quantiles for each scalar distribution.
  """
  try:
    tail = distribution._upper_tail(tail_mass)  # pylint:disable=protected-access
  except (AttributeError, NotImplementedError):
    try:
      tail = distribution.quantile(1 - tail_mass / 2)
    except NotImplementedError:
      try:
        target = tf.math.log(tf.cast(tail_mass / 2, distribution.dtype))
        tail = estimate_tails(
            distribution.log_survival_function, target,
            distribution.batch_shape_tensor(), distribution.dtype)
      except NotImplementedError:
        raise NotImplementedError(
            "`distribution` must implement `_upper_tail()`, `quantile()`, or "
            "`log_survival_function()` so that upper tail can be located.")
  return tf.stop_gradient(tail)
