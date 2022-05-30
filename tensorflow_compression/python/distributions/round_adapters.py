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
"""Distribution adapters for (soft) round functions."""

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_compression.python.distributions import deep_factorized
from tensorflow_compression.python.distributions import helpers
from tensorflow_compression.python.distributions import uniform_noise
from tensorflow_compression.python.ops import round_ops


__all__ = [
    "MonotonicAdapter",
    "RoundAdapter",
    "NoisyRoundedNormal",
    "NoisyRoundedDeepFactorized",
    "SoftRoundAdapter",
    "NoisySoftRoundedNormal",
    "NoisySoftRoundedDeepFactorized",
]


class MonotonicAdapter(tfp.distributions.Distribution):
  """Adapt a continuous distribution via an ascending monotonic function.

  This is described in Appendix E. in the paper
  > "Universally Quantized Neural Compression"<br />
  > Eirikur Agustsson & Lucas Theis<br />
  > https://arxiv.org/abs/2006.09952

  """

  invertible = True  # Set to false if the transform is not invertible.

  def __init__(self, base, name="MonotonicAdapter"):
    """Initializer.

    Args:
      base: A `tfp.distributions.Distribution` object representing a
        continuous-valued random variable.
      name: String. A name for this distribution.
    """
    parameters = dict(locals())
    self._base = base
    super().__init__(
        dtype=base.dtype,
        reparameterization_type=base.reparameterization_type,
        validate_args=base.validate_args,
        allow_nan_stats=base.allow_nan_stats,
        parameters=parameters,
        name=name,
    )

  @property
  def base(self):
    """The base distribution."""
    return self._base

  def transform(self, x):
    """The forward transform."""
    raise NotImplementedError()

  def inverse_transform(self, y):
    """The backward transform."""
    # Let f(x) = self.transform(x)
    # Then g(y) = self.inverse_transform(y) is defined as
    # g(y) := inf_x { x : f(x) >= y }
    # which is just the inverse of `f` if it is invertible.
    raise NotImplementedError()

  def _batch_shape_tensor(self):
    return self.base.batch_shape_tensor()

  def _batch_shape(self):
    return self.base.batch_shape

  def _event_shape_tensor(self):
    return self.base.event_shape_tensor()

  def _event_shape(self):
    return self.base.event_shape

  def _sample_n(self, n, seed=None):
    with tf.name_scope("round"):
      n = tf.convert_to_tensor(n, name="n")
      samples = self.base.sample(n, seed=seed)
      return self.transform(samples)

  def _prob(self, *args, **kwargs):
    raise NotImplementedError

  def _log_prob(self, *args, **kwargs):
    raise NotImplementedError

  # pylint: disable=protected-access
  def _cdf(self, y):
    # Let f be the forward transform and g the inverse.
    # Then we have:
    #   P( f(x) <= y )
    #   P( g(f(x)) <= g(y) )
    # = P(  x <= g(y) )
    return self.base._cdf(self.inverse_transform(y))

  def _log_cdf(self, y):
    return self.base._log_cdf(self.inverse_transform(y))

  def _survival_function(self, y):
    return self.base._survival_function(self.inverse_transform(y))

  def _log_survival_function(self, y):
    return self.base._log_survival_function(self.inverse_transform(y))

  def _quantile(self, value):
    if not self.invertible:
      raise NotImplementedError()
    # We have:
    # P( x <= z ) = value
    #   if and only if
    # P( f(x) <= f(z) ) = value
    return self.transform(self.base._quantile(value))

  def _mode(self):
    # Same logic as for _quantile.
    if not self.invertible:
      raise NotImplementedError()
    return self.transform(self.base._mode())

  def _quantization_offset(self):
    # Same logic as for _quantile.
    if not self.invertible:
      raise NotImplementedError()
    return self.transform(helpers.quantization_offset(self.base))

  def _lower_tail(self, tail_mass):
    # Same logic as for _quantile.
    if not self.invertible:
      raise NotImplementedError()
    return self.transform(helpers.lower_tail(self.base, tail_mass))

  def _upper_tail(self, tail_mass):
    # Same logic as for _quantile.
    if not self.invertible:
      raise NotImplementedError()
    return self.transform(helpers.upper_tail(self.base, tail_mass))
  # pylint: enable=protected-access

  @classmethod
  def _parameter_properties(cls, dtype=tf.float32, num_classes=None):
    raise NotImplementedError(
        f"`{cls.__name__}` does not implement `_parameter_properties`.")


class RoundAdapter(MonotonicAdapter):
  """Continuous density function + round."""

  invertible = False

  def transform(self, x):
    return tf.round(x)

  def inverse_transform(self, y):
    # Let f(x) = round(x)
    # Then g(y) = inverse_transform(y) is defined as
    # g(y) := inf_x { x : f(x) >= y }
    # For f = round, we have
    #     round(x) >= y
    # <=> round(x) >= ceil(y)
    # so g(y) = inf_x { x: round(x) >= ceil(y) }
    #         = ceil(y)-0.5

    # Alternative derivation:
    # P( round(x) <= y )
    # = P( round(x) <= floor(y) )
    # = P( x <= floor(y)+0.5 )
    # = P( x <= ceil(y)-0.5 )
    # = P( x <= inverse_transform(y) )
    return tf.math.ceil(y) - 0.5

  def _quantization_offset(self):
    return tf.convert_to_tensor(0.0, dtype=self.dtype)

  def _lower_tail(self, tail_mass):
    return tf.math.floor(helpers.lower_tail(self.base, tail_mass))

  def _upper_tail(self, tail_mass):
    return tf.math.ceil(helpers.upper_tail(self.base, tail_mass))


class NoisyRoundAdapter(uniform_noise.UniformNoiseAdapter):
  """Uniform noise + round."""

  def __init__(self, base, name="NoisyRoundAdapter"):
    """Initializer.

    Args:
      base: A `tfp.distributions.Distribution` object representing a
        continuous-valued random variable.
      name: String. A name for this distribution.
    """
    super().__init__(RoundAdapter(base), name=name)


class NoisyRoundedDeepFactorized(NoisyRoundAdapter):
  """Rounded `DeepFactorized` + uniform noise."""

  def __init__(self, name="NoisyRoundedDeepFactorized", **kwargs):
    prior = deep_factorized.DeepFactorized(**kwargs)
    super().__init__(base=prior, name=name)


class NoisyRoundedNormal(NoisyRoundAdapter):
  """Rounded normal distribution + uniform noise."""

  def __init__(self, name="NoisyRoundedNormal", **kwargs):
    super().__init__(base=tfp.distributions.Normal(**kwargs), name=name)


class SoftRoundAdapter(MonotonicAdapter):
  """Differentiable approximation to round."""

  def __init__(self, base, alpha, name="SoftRoundAdapter"):
    """Initializer.

    Args:
      base: A `tfp.distributions.Distribution` object representing a
        continuous-valued random variable.
      alpha: Float or tf.Tensor. Controls smoothness of the approximation.
      name: String. A name for this distribution.
    """
    super().__init__(base=base, name=name)
    self._alpha = alpha

  def transform(self, x):
    return round_ops.soft_round(x, self._alpha)

  def inverse_transform(self, y):
    return round_ops.soft_round_inverse(y, self._alpha)


class NoisySoftRoundAdapter(uniform_noise.UniformNoiseAdapter):
  """Uniform noise + differentiable approximation to round."""

  def __init__(self, base, alpha, name="NoisySoftRoundAdapter"):
    """Initializer.

    Args:
      base: A `tfp.distributions.Distribution` object representing a
        continuous-valued random variable.
      alpha: Float or tf.Tensor. Controls smoothness of soft round.
      name: String. A name for this distribution.
    """
    super().__init__(SoftRoundAdapter(base, alpha), name=name)


class NoisySoftRoundedNormal(NoisySoftRoundAdapter):
  """Soft rounded normal distribution + uniform noise."""

  def __init__(self, alpha=5.0, name="NoisySoftRoundedNormal", **kwargs):
    super().__init__(
        base=tfp.distributions.Normal(**kwargs),
        alpha=alpha,
        name=name)


class NoisySoftRoundedDeepFactorized(NoisySoftRoundAdapter):
  """Soft rounded `DeepFactorized` + uniform noise."""

  def __init__(self,
               alpha=5.0,
               name="NoisySoftRoundedDeepFactorized",
               **kwargs):
    super().__init__(
        base=deep_factorized.DeepFactorized(**kwargs),
        alpha=alpha,
        name=name)
