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
"""Extensions to `tfp.distributions.Distribution` to handle range coding."""

import tensorflow.compat.v2 as tf


__all__ = [
    "quantization_offset",
    "lower_tail",
    "upper_tail",
]


def estimate_tail(func, target, shape, dtype):
  """Estimates approximate tail quantiles."""
  dtype = tf.as_dtype(dtype)
  shape = tf.convert_to_tensor(shape, tf.int32)
  target = tf.convert_to_tensor(target, dtype)
  opt = tf.keras.optimizers.Adam(learning_rate=.1)
  tails = tf.Variable(
      tf.zeros(shape, dtype=dtype), trainable=False, name="tails")
  loss = best_loss = tf.fill(shape, tf.constant(float("inf"), dtype=dtype))
  while tf.reduce_any(loss == best_loss):
    with tf.GradientTape(watch_accessed_variables=False) as tape:
      tape.watch(tails)
      loss = abs(func(tails) - target)
    best_loss = tf.minimum(best_loss, loss)
    gradient = tape.gradient(loss, tails)
    opt.apply_gradients([(gradient, tails)])
  return tails.value()


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

  Arguments:
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
  return tf.stop_gradient(offset)


def lower_tail(distribution, tail_mass):
  """Approximates lower tail quantile for range coding.

  For range coding of random variables, the distribution tails need special
  handling, because range coding can only handle alphabets with a finite
  number of symbols. This method returns a cut-off location for the lower
  tail, such that approximately `tail_mass` probability mass is contained in
  the tails (together). The tails are then handled by using the 'overflow'
  functionality of the range coder implementation (using a Golomb-like
  universal code).

  Arguments:
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
        tail = estimate_tail(
            distribution.log_cdf, tf.math.log(tail_mass / 2),
            distribution.batch_shape_tensor(), distribution.dtype)
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
  functionality of the range coder implementation (using a Golomb-like
  universal code).

  Arguments:
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
        tail = estimate_tail(
            distribution.log_survival_function, tf.math.log(tail_mass / 2),
            distribution.batch_shape_tensor(), distribution.dtype)
      except NotImplementedError:
        raise NotImplementedError(
            "`distribution` must implement `_upper_tail()`, `quantile()`, or "
            "`log_survival_function()` so that upper tail can be located.")
  return tf.stop_gradient(tail)
