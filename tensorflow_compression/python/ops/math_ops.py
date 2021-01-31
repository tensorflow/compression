# Copyright 2018 Google LLC. All Rights Reserved.
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
"""Math operations."""

import tensorflow as tf


__all__ = [
    "upper_bound",
    "lower_bound",
    "perturb_and_apply",
]


def upper_bound(inputs, bound, gradient="identity_if_towards",
                name="upper_bound"):
  """Same as `tf.minimum`, but with helpful gradient for `inputs > bound`.

  This function behaves just like `tf.minimum`, but the behavior of the gradient
  with respect to `inputs` for input values that hit the bound depends on
  `gradient`:

  If set to `'disconnected'`, the returned gradient is zero for values that hit
  the bound. This is identical to the behavior of `tf.minimum`.

  If set to `'identity'`, the gradient is unconditionally replaced with the
  identity function (i.e., pretending this function does not exist).

  If set to `'identity_if_towards'`, the gradient is replaced with the identity
  function, but only if applying gradient descent would push the values of
  `inputs` towards the bound. For gradient values that push away from the bound,
  the returned gradient is still zero.

  Note: In the latter two cases, no gradient is returned for `bound`.
  Also, the implementation of `gradient == 'identity_if_towards'` currently
  assumes that the shape of `inputs` is the same as the shape of the output. It
  won't work reliably for all possible broadcasting scenarios.

  Args:
    inputs: Input tensor.
    bound: Upper bound for the input tensor.
    gradient: 'disconnected', 'identity', or 'identity_if_towards' (default).
    name: Name for this op.

  Returns:
    `tf.minimum(inputs, bound)`

  Raises:
    ValueError: for invalid value of `gradient`.
  """
  with tf.name_scope(name) as scope:
    inputs = tf.convert_to_tensor(inputs, name="inputs")
    bound = tf.convert_to_tensor(bound, name="bound", dtype=inputs.dtype)

    def identity_if_towards_grad(grad):
      """Gradient if gradient == 'identity_if_towards'."""
      pass_through_if = tf.logical_or(inputs <= bound, grad > 0)
      return (tf.cast(pass_through_if, grad.dtype) * grad, None)

    def disconnected_grad(grad):
      """Gradient if gradient == 'disconnected'."""
      return (tf.cast(inputs <= bound, grad.dtype) * grad, None)

    try:
      gradient = {
          "identity_if_towards": identity_if_towards_grad,
          "identity": lambda grad: (grad, None),
          "disconnected": disconnected_grad,
      }[gradient]
    except KeyError:
      raise ValueError("Invalid value for `gradient`: '{}'.".format(gradient))

    @tf.custom_gradient
    def _upper_bound(inputs, bound):
      return tf.minimum(inputs, bound, name=scope), gradient

    return _upper_bound(inputs, bound)


def lower_bound(inputs, bound, gradient="identity_if_towards",
                name="lower_bound"):
  """Same as `tf.maximum`, but with helpful gradient for `inputs < bound`.

  This function behaves just like `tf.maximum`, but the behavior of the gradient
  with respect to `inputs` for input values that hit the bound depends on
  `gradient`:

  If set to `'disconnected'`, the returned gradient is zero for values that hit
  the bound. This is identical to the behavior of `tf.maximum`.

  If set to `'identity'`, the gradient is unconditionally replaced with the
  identity function (i.e., pretending this function does not exist).

  If set to `'identity_if_towards'`, the gradient is replaced with the identity
  function, but only if applying gradient descent would push the values of
  `inputs` towards the bound. For gradient values that push away from the bound,
  the returned gradient is still zero.

  Note: In the latter two cases, no gradient is returned for `bound`.
  Also, the implementation of `gradient == 'identity_if_towards'` currently
  assumes that the shape of `inputs` is the same as the shape of the output. It
  won't work reliably for all possible broadcasting scenarios.

  Args:
    inputs: Input tensor.
    bound: Lower bound for the input tensor.
    gradient: 'disconnected', 'identity', or 'identity_if_towards' (default).
    name: Name for this op.

  Returns:
    `tf.maximum(inputs, bound)`

  Raises:
    ValueError: for invalid value of `gradient`.
  """
  with tf.name_scope(name) as scope:
    inputs = tf.convert_to_tensor(inputs, name="inputs")
    bound = tf.convert_to_tensor(bound, name="bound", dtype=inputs.dtype)

    def identity_if_towards_grad(grad):
      """Gradient if gradient == 'identity_if_towards'."""
      pass_through_if = tf.logical_or(inputs >= bound, grad < 0)
      return (tf.cast(pass_through_if, grad.dtype) * grad, None)

    def disconnected_grad(grad):
      """Gradient if gradient == 'disconnected'."""
      return (tf.cast(inputs >= bound, grad.dtype) * grad, None)

    try:
      gradient = {
          "identity_if_towards": identity_if_towards_grad,
          "identity": lambda grad: (grad, None),
          "disconnected": disconnected_grad,
      }[gradient]
    except KeyError:
      raise ValueError("Invalid value for `gradient`: '{}'.".format(gradient))

    @tf.custom_gradient
    def _lower_bound(inputs, bound):
      return tf.maximum(inputs, bound, name=scope), gradient

    return _lower_bound(inputs, bound)


def perturb_and_apply(f, x, *args, u=None, x_plus_u=None, expected_grads=True):
  """Perturbs the inputs of a pointwise function.

  This function adds uniform noise in the range -0.5 to 0.5 to the first
  argument of the given function.
  It further replaces derivatives of the function with (analytically computed)
  expected derivatives w.r.t. the noise.

  This is described in Sec. 4.2. in the paper
  > "Universally Quantized Neural Compression"<br />
  > Eirikur Agustsson & Lucas Theis<br />
  > https://arxiv.org/abs/2006.09952

  Args:
    f: Callable. Pointwise function applied after perturbation.
    x: The inputs.
    *args: Other arguments to f.
    u: The noise to perturb x with. If not set and x_plus_u is not provided,
      it will be sampled.
    x_plus_u: Alternative way to provide the noise, as `x+u`.
    expected_grads: If True, will compute expected gradients.

  Returns:
   A tuple (y, x+u) where y=f(x+u, *args) and u is uniform noise, and the
   gradient of `y` w.r.t. `x` uses expected derivatives w.r.t. the distribution
   of u.
  """

  if x_plus_u is None:
    if u is None:
      u = tf.random.uniform(tf.shape(x), minval=-.5, maxval=.5, dtype=x.dtype)
    x_plus_u = x + u
  elif u is not None:
    raise ValueError("Cannot provide both `u` and `x_plus_u`.")

  args = [tf.convert_to_tensor(arg) for arg in args]
  if not expected_grads:
    return f(x_plus_u, *args), x_plus_u

  @tf.custom_gradient
  def _perturb_and_apply(x, *args):
    tape = tf.GradientTape(persistent=True)
    with tape:
      tape.watch(args)
      y = f(x_plus_u, *args)

    # The expected derivative of f with respect to x
    dydx = f(x + 0.5, *args) - f(x - 0.5, *args)

    def grad(grad_ys, variables=None):
      # Calculate gradients of other variables as usual
      if variables is None:
        variables = []
      grad_args, grad_vars = tape.gradient(
          y, (args, variables), output_gradients=grad_ys)
      return [grad_ys * dydx] + list(grad_args), grad_vars

    return y, grad

  return _perturb_and_apply(x, *args), x_plus_u
