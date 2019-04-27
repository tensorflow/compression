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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


__all__ = [
    "upper_bound",
    "lower_bound",
]


@tf.RegisterGradient("IdentityFirstOfTwoInputs")
def _identity_first_of_two_inputs_grad(op, grad):
  """Gradient for `lower_bound` or `upper_bound` if `gradient == 'identity'`.

  Args:
    op: The op for which to calculate a gradient.
    grad: Gradient with respect to the output of the op.

  Returns:
    Gradient with respect to the inputs of the op.
  """
  del op  # unused
  return [grad, None]


@tf.RegisterGradient("UpperBound")
def _upper_bound_grad(op, grad):
  """Gradient for `upper_bound` if `gradient == 'identity_if_towards'`.

  Args:
    op: The op for which to calculate a gradient.
    grad: Gradient with respect to the output of the op.

  Returns:
    Gradient with respect to the inputs of the op.
  """
  inputs, bound = op.inputs
  pass_through_if = tf.logical_or(inputs <= bound, grad > 0)
  return [tf.cast(pass_through_if, grad.dtype) * grad, None]


@tf.RegisterGradient("LowerBound")
def _lower_bound_grad(op, grad):
  """Gradient for `lower_bound` if `gradient == 'identity_if_towards'`.

  Args:
    op: The op for which to calculate a gradient.
    grad: Gradient with respect to the output of the op.

  Returns:
    Gradient with respect to the inputs of the op.
  """
  inputs, bound = op.inputs
  pass_through_if = tf.logical_or(inputs >= bound, grad < 0)
  return [tf.cast(pass_through_if, grad.dtype) * grad, None]


def upper_bound(inputs, bound, gradient="identity_if_towards", name=None):
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
  try:
    gradient = {
        "identity_if_towards": "UpperBound",
        "identity": "IdentityFirstOfTwoInputs",
        "disconnected": None,
    }[gradient]
  except KeyError:
    raise ValueError("Invalid value for `gradient`: '{}'.".format(gradient))

  with tf.name_scope(name, "UpperBound", [inputs, bound]) as scope:
    inputs = tf.convert_to_tensor(inputs, name="inputs")
    bound = tf.convert_to_tensor(
        bound, name="bound", dtype=inputs.dtype)
    if gradient:
      with tf.get_default_graph().gradient_override_map({"Minimum": gradient}):
        return tf.minimum(inputs, bound, name=scope)
    else:
      return tf.minimum(inputs, bound, name=scope)


def lower_bound(inputs, bound, gradient="identity_if_towards", name=None):
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
  try:
    gradient = {
        "identity_if_towards": "LowerBound",
        "identity": "IdentityFirstOfTwoInputs",
        "disconnected": None,
    }[gradient]
  except KeyError:
    raise ValueError("Invalid value for `gradient`: '{}'.".format(gradient))

  with tf.name_scope(name, "LowerBound", [inputs, bound]) as scope:
    inputs = tf.convert_to_tensor(inputs, name="inputs")
    bound = tf.convert_to_tensor(
        bound, name="bound", dtype=inputs.dtype)
    if gradient:
      with tf.get_default_graph().gradient_override_map({"Maximum": gradient}):
        return tf.maximum(inputs, bound, name=scope)
    else:
      return tf.maximum(inputs, bound, name=scope)
