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
"""Uniform noise adapter distribution."""

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_compression.python.distributions import helpers


__all__ = [
    "UniformNoiseAdapter",
    "NoisyMixtureSameFamily",
    "NoisyNormal",
    "NoisyLogistic",
    "NoisyLaplace",
    "NoisyNormalMixture",
    "NoisyLogisticMixture",
]


def _logsum_expbig_minus_expsmall(big, small):
  """Numerically stable evaluation of `log(exp(big) - exp(small))`.

  This assumes `small <= big` and arguments that can be broadcast against each
  other.

  Args:
    big: Floating-point `tf.Tensor`.
    small: Floating-point `tf.Tensor`.

  Returns:
    `tf.Tensor` containing the result.
  """
  with tf.name_scope("logsum_expbig_minus_expsmall"):
    # Have to special case `inf` and `-inf` since otherwise we get a NaN
    # out of the exp (if both small and big are -inf).
    return tf.where(
        tf.math.is_inf(big), big, tf.math.log1p(-tf.exp(small - big)) + big
    )


class UniformNoiseAdapter(tfp.distributions.Distribution):
  """Additive i.i.d. uniform noise adapter distribution.

  Given a base `tfp.distributions.Distribution` object, this distribution
  models the base distribution after addition of independent uniform noise.

  Effectively, the base density function is convolved with a box kernel of width
  one. The resulting density can be efficiently evaluated via the relation:
  ```
  (p * u)(x) = c(x + .5) - c(x - .5)
  ```
  where `p` and `u` are the base density and the unit-width uniform density,
  respectively, and `c` is the cumulative distribution function (CDF)
  corresponding to `p`. This is described in appendix 6.2 of the paper:

  > "Variational image compression with a scale hyperprior"<br />
  > J. Ballé, D. Minnen, S. Singh, S. J. Hwang, N. Johnston<br />
  > https://openreview.net/forum?id=rkcQFMZRb

  For best numerical stability, the base `Distribution` should implement both
  `cdf()` and `survival_function()` and/or their `log_*` equivalents.
  """

  def __init__(self, base, name="UniformNoiseAdapter"):
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
    """The base distribution (without uniform noise)."""
    return self._base

  def _batch_shape_tensor(self):
    return self.base.batch_shape_tensor()

  def _batch_shape(self):
    return self.base.batch_shape

  def _event_shape_tensor(self):
    return self.base.event_shape_tensor()

  def _event_shape(self):
    return self.base.event_shape

  def _sample_n(self, n, seed=None):
    with tf.name_scope("transform"):
      n = tf.convert_to_tensor(n, name="n")
      samples = self.base.sample(n, seed=seed)
      return samples + tf.random.uniform(
          tf.shape(samples), minval=-.5, maxval=.5, dtype=samples.dtype)

  def _log_prob(self, y):
    if not hasattr(self.base, "_log_cdf"):
      raise NotImplementedError(
          "`log_prob()` is not implemented unless the base distribution "
          "implements `log_cdf()`.")
    try:
      return self._log_prob_with_logsf_and_logcdf(y)
    except NotImplementedError:
      return self._log_prob_with_logcdf(y)

  def _log_prob_with_logcdf(self, y):
    return _logsum_expbig_minus_expsmall(
        self.base.log_cdf(y + .5), self.base.log_cdf(y - .5))

  def _log_prob_with_logsf_and_logcdf(self, y):
    """Compute log_prob(y) using log survival_function and cdf together."""
    # There are two options that would be equal if we had infinite precision:
    # Log[ sf(y - .5) - sf(y + .5) ]
    #   = Log[ exp{logsf(y - .5)} - exp{logsf(y + .5)} ]
    # Log[ cdf(y + .5) - cdf(y - .5) ]
    #   = Log[ exp{logcdf(y + .5)} - exp{logcdf(y - .5)} ]
    logsf_y_plus = self.base.log_survival_function(y + .5)
    logsf_y_minus = self.base.log_survival_function(y - .5)
    logcdf_y_plus = self.base.log_cdf(y + .5)
    logcdf_y_minus = self.base.log_cdf(y - .5)

    # Important:  Here we use select in a way such that no input is inf, this
    # prevents the troublesome case where the output of select can be finite,
    # but the output of grad(select) will be NaN.

    # In either case, we are doing Log[ exp{big} - exp{small} ]
    # We want to use the sf items precisely when we are on the right side of the
    # median, which occurs when logsf_y < logcdf_y.
    condition = logsf_y_plus < logcdf_y_plus
    big = tf.where(condition, logsf_y_minus, logcdf_y_plus)
    small = tf.where(condition, logsf_y_plus, logcdf_y_minus)
    return _logsum_expbig_minus_expsmall(big, small)

  def _prob(self, y):
    if not hasattr(self.base, "_cdf"):
      raise NotImplementedError(
          "`prob()` is not implemented unless the base distribution implements "
          "`cdf()`.")
    try:
      return self._prob_with_sf_and_cdf(y)
    except NotImplementedError:
      return self._prob_with_cdf(y)

  def _prob_with_cdf(self, y):
    return self.cdf(y + .5) - self.cdf(y - .5)

  def _prob_with_sf_and_cdf(self, y):
    # There are two options that would be equal if we had infinite precision:
    # sf(y - .5) - sf(y + .5)
    # cdf(y + .5) - cdf(y - .5)
    sf_y_plus = self.base.survival_function(y + .5)
    sf_y_minus = self.base.survival_function(y - .5)
    cdf_y_plus = self.base.cdf(y + .5)
    cdf_y_minus = self.base.cdf(y - .5)

    # sf_prob has greater precision iff we're on the right side of the median.
    return tf.where(
        sf_y_plus < cdf_y_plus,
        sf_y_minus - sf_y_plus, cdf_y_plus - cdf_y_minus)

  def _mean(self):
    return self.base.mean()

  def _quantization_offset(self):
    return helpers.quantization_offset(self.base)

  def _lower_tail(self, tail_mass):
    return helpers.lower_tail(self.base, tail_mass)

  def _upper_tail(self, tail_mass):
    return helpers.upper_tail(self.base, tail_mass)

  @classmethod
  def _parameter_properties(cls, dtype=tf.float32, num_classes=None):
    raise NotImplementedError(
        f"`{cls.__name__}` does not implement `_parameter_properties`.")


class NoisyMixtureSameFamily(tfp.distributions.MixtureSameFamily):
  """Mixture of distributions with additive i.i.d. uniform noise."""

  def __init__(self, mixture_distribution, components_distribution,
               name="NoisyMixtureSameFamily"):
    """Initializer, taking the same arguments as `tfpd.MixtureSameFamily`."""
    super().__init__(
        mixture_distribution=mixture_distribution,
        components_distribution=UniformNoiseAdapter(components_distribution),
        name=name,
    )
    self._base = tfp.distributions.MixtureSameFamily(
        mixture_distribution=mixture_distribution,
        components_distribution=components_distribution,
        name=name + "Base",
    )

  @property
  def base(self):
    """The base distribution (without uniform noise)."""
    return self._base

  def _batch_shape_tensor(self):
    return self.base.batch_shape_tensor()

  def _batch_shape(self):
    return self.base.batch_shape

  def _event_shape_tensor(self):
    return self.base.event_shape_tensor()

  def _event_shape(self):
    return self.base.event_shape

  def _quantization_offset(self):
    # Picks the "peakiest" of the component quantization offsets.
    offsets = helpers.quantization_offset(self.components_distribution)
    rank = self.batch_shape.rank
    transposed_offsets = tf.transpose(offsets, [rank] + list(range(rank)))
    component = tf.argmax(self.log_prob(transposed_offsets), axis=0)
    return tf.gather(offsets, component, axis=-1, batch_dims=rank)

  def _lower_tail(self, tail_mass):
    return helpers.lower_tail(self.base, tail_mass)

  def _upper_tail(self, tail_mass):
    return helpers.upper_tail(self.base, tail_mass)

  @classmethod
  def _parameter_properties(cls, dtype=tf.float32, num_classes=None):
    raise NotImplementedError(
        f"`{cls.__name__}` does not implement `_parameter_properties`.")


class NoisyNormal(UniformNoiseAdapter):
  """Gaussian distribution with additive i.i.d. uniform noise."""

  def __init__(self, name="NoisyNormal", **kwargs):
    """Initializer, taking the same arguments as `tfpd.Normal`."""
    super().__init__(tfp.distributions.Normal(**kwargs), name=name)


class NoisyLogistic(UniformNoiseAdapter):
  """Logistic distribution with additive i.i.d. uniform noise."""

  def __init__(self, name="NoisyLogistic", **kwargs):
    """Initializer, taking the same arguments as `tfpd.Logistic`."""
    super().__init__(tfp.distributions.Logistic(**kwargs), name=name)


class NoisyLaplace(UniformNoiseAdapter):
  """Laplacian distribution with additive i.i.d. uniform noise."""

  def __init__(self, name="NoisyLaplace", **kwargs):
    """Initializer, taking the same arguments as `tfpd.Laplace`."""
    super().__init__(tfp.distributions.Laplace(**kwargs), name=name)


class NoisyNormalMixture(NoisyMixtureSameFamily):
  """Mixture of normal distributions with additive i.i.d. uniform noise."""

  def __init__(self, loc, scale, weight, name="NoisyNormalMixture"):
    """Initializer.

    Args:
      loc: Location parameters of `tfpd.Normal` component distributions.
      scale: Scale parameters of `tfpd.Normal` component distributions.
      weight: `probs` parameter of `tfpd.Categorical` mixture distribution.
      name: A name for this distribution.
    """
    super().__init__(
        mixture_distribution=tfp.distributions.Categorical(probs=weight),
        components_distribution=tfp.distributions.Normal(loc=loc, scale=scale),
        name=name,
    )


class NoisyLogisticMixture(NoisyMixtureSameFamily):
  """Mixture of logistic distributions with additive i.i.d. uniform noise."""

  def __init__(self, loc, scale, weight, name="NoisyLogisticMixture"):
    """Initializer.

    Args:
      loc: Location parameters of `tfpd.Logistic` component distributions.
      scale: Scale parameters of `tfpd.Logistic` component distributions.
      weight: `probs` parameter of `tfpd.Categorical` mixture distribution.
      name: A name for this distribution.
    """
    super().__init__(
        mixture_distribution=tfp.distributions.Categorical(probs=weight),
        components_distribution=tfp.distributions.Logistic(
            loc=loc, scale=scale),
        name=name,
    )
