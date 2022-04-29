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
"""Nonlinear transform coding."""

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_compression as tfc
import tensorflow_probability as tfp
from toy_sources import compression_model

tfpd = tfp.distributions


class NTCModel(compression_model.CompressionModel):
  """Nonlinear transform coding model."""

  def __init__(self, analysis, synthesis, prior_type="deep",
               dither=(1, 1, 0, 0), soft_round=(1, 0), guess_offset=False,
               **kwargs):
    """Initializer.

    Args:
      analysis: A `Layer` object implementing the analysis transform.
      synthesis: A `Layer` object implementing the synthesis transform.
      prior_type: String. Either 'deep' for `DeepFactorized` prior, or
        'gsm/gmm/lsm/lmm-X' for Gaussian/Logistic Scale Mixture/Mixture Model
        with X components.
      dither: Sequence of 4 Booleans. Whether to use dither for: rate term
        during training, distortion term during training, rate term during
        testing, distortion term during testing, respectively.
      soft_round: Sequence of 2 Booleans. Whether to use soft rounding during
        training or testing, respectively.
      guess_offset: Boolean. When not using soft rounding, whether to use the
        mode centering heuristic to determine the quantization offset during
        testing.
      **kwargs: Other arguments passed through to `CompressionModel` class.
    """
    super().__init__(**kwargs)
    self._analysis = analysis
    self._synthesis = synthesis
    self.prior_type = str(prior_type)
    self.dither = tuple(bool(i) for i in dither)
    self.soft_round = tuple(bool(i) for i in soft_round)
    self.guess_offset = bool(guess_offset)

    if self.prior_type == "deep":
      self._prior = tfc.DeepFactorized(
          batch_shape=[self.ndim_latent], dtype=self.dtype)
    elif self.prior_type[:4] in ("gsm-", "gmm-", "lsm-", "lmm-"):
      components = int(self.prior_type[4:])
      shape = (self.ndim_latent, components)
      self.logits = tf.Variable(tf.random.normal(shape, dtype=self.dtype))
      self.log_scale = tf.Variable(
          tf.random.normal(shape, mean=2., dtype=self.dtype))
      if "s" in self.prior_type:
        self.loc = 0.
      else:
        self.loc = tf.Variable(tf.random.normal(shape, dtype=self.dtype))
    else:
      raise ValueError(f"Unknown prior_type: '{prior_type}'.")

    self._logit_alpha = tf.Variable(-3, dtype=self.dtype, name="logit_alpha")
    self._force_alpha = tf.Variable(
        -1, trainable=False, dtype=self.dtype, name="force_alpha")

  def prior(self, soft_round, scale=None, alpha=None, skip_noise=False):
    if self.prior_type == "deep":
      prior = self._prior
    elif self.prior_type[:4] in ("gsm-", "gmm-", "lsm-", "lmm-"):
      cls = tfpd.Normal if self.prior_type.startswith("g") else tfpd.Logistic
      prior = tfpd.MixtureSameFamily(
          mixture_distribution=tfpd.Categorical(logits=self.logits),
          components_distribution=cls(
              loc=self.loc, scale=tf.math.exp(self.log_scale)),
      )
    if soft_round:
      if alpha is None:
        alpha = self.alpha
      prior = tfc.SoftRoundAdapter(prior, alpha)
    if skip_noise:
      return prior
    return tfc.UniformNoiseAdapter(prior)

  @property
  def ndim_latent(self):
    return self._analysis.output_shape[-1]

  def analysis(self, x):
    y = tf.cast(x, self.dtype)
    if y.shape[-1] != self.ndim_source:
      raise ValueError(
          f"Expected {self.ndim_source} trailing dimensions, "
          f"received {y.shape[-1]}.")
    batch_shape = tf.shape(y)[:-1]
    y = tf.reshape(y, (-1, self.ndim_source))
    y = self._analysis(y)
    assert y.shape[-1] == self.ndim_latent
    return tf.reshape(y, tf.concat([batch_shape, [self.ndim_latent]], 0))

  def synthesis(self, y):
    x = tf.cast(y, self.dtype)
    if x.shape[-1] != self.ndim_latent:
      raise ValueError(
          f"Expected {self.ndim_latent} trailing dimensions, "
          f"received {x.shape[-1]}.")
    batch_shape = tf.shape(x)[:-1]
    x = tf.reshape(x, (-1, self.ndim_latent))
    x = self._synthesis(x)
    assert x.shape[-1] == self.ndim_source
    return tf.reshape(x, tf.concat([batch_shape, [self.ndim_source]], 0))

  @property
  def force_alpha(self):
    return tf.convert_to_tensor(self._force_alpha)

  @force_alpha.setter
  def force_alpha(self, value):
    if value is None:
      value = -1.
    self._force_alpha.assign(value)

  @property
  def alpha(self):
    return tf.math.sigmoid(self._logit_alpha) * 4.

  @alpha.setter
  def alpha(self, value):
    value = tf.convert_to_tensor(value, self.dtype)

    def get_logit_alpha():
      a = tf.clip_by_value(value / 4., 0., 1.)
      logit_alpha = tf.math.log(a / (1. - a))
      return logit_alpha

    self._logit_alpha.assign(
        tf.cond(value < 0, lambda: self._logit_alpha, get_logit_alpha))

  def encode_decode(self, x, dither_rate, dither_dist, soft_round,
                    guess_offset=None, offset=0., seed=None):
    if guess_offset is None:
      guess_offset = self.guess_offset
    # It doesn't make sense to use both guess_offset and soft_round.
    assert not (guess_offset and soft_round)

    def perturb(inputs, dither, prior, offset):
      if dither:
        if soft_round:
          inputs = tfc.soft_round(inputs, alpha=self.alpha)
        inputs += tf.random.uniform(
            tf.shape(inputs), -.5, .5, dtype=self.dtype, seed=seed)
        if soft_round:
          inputs = tfc.soft_round_conditional_mean(inputs, alpha=self.alpha)
        return inputs
      else:
        if guess_offset:
          offset += tfc.quantization_offset(prior)
        return tfc.round_st(inputs, offset)

    assert x.shape[-1] == self.ndim_source
    y = self.analysis(x)

    rates = 0.
    prior = self.prior(soft_round=soft_round)

    y_dist = perturb(y, dither_dist, prior, offset)
    if dither_rate == dither_dist:
      y_rate = y_dist
    else:
      y_rate = perturb(y, dither_rate, prior, offset)

    x_hat = self.synthesis(y_dist)
    log_probs = prior.log_prob(y_rate)
    rates += tf.reduce_sum(log_probs, axis=-1) / tf.cast(
        -tf.math.log(2.), self.dtype)

    return y_dist, x_hat, rates

  def quantize(self, x, **kwargs):
    y_hat, x_hat, rates = self.encode_decode(x, False, False, False, **kwargs)
    # Find the unique set of latents for these inputs. Converts integer indexes
    # on the infinite lattice to scalar indexes into a codebook (which is only
    # valid for this set of inputs).
    _, i, indexes = np.unique(
        tf.reshape(y_hat, [-1, self.ndim_latent]),
        return_index=True, return_inverse=True, axis=0)
    codebook = tf.gather(tf.reshape(x_hat, [-1, self.ndim_source]), i, axis=0)
    rates = tf.gather(tf.reshape(rates, [-1]), i)
    indexes = tf.reshape(tf.cast(indexes, tf.int32), tf.shape(x)[:-1])
    return codebook, rates, indexes

  def train_losses(self, x):
    _, x_hat, rates = self.encode_decode(
        x, self.dither[0], self.dither[1], self.soft_round[0])
    distortions = self.distortion_fn(x, x_hat)
    return rates, distortions

  def test_losses(self, x):
    _, x_hat, rates = self.encode_decode(
        x, self.dither[2], self.dither[3], self.soft_round[1])
    distortions = self.distortion_fn(x, x_hat)
    return rates, distortions

  def plot_transfer(self, intervals, figsize=None, soft_round=None, **kwargs):
    if not len(intervals) == self.ndim_source == self.ndim_latent == 1:
      raise ValueError("This method is only defined for 1D models.")
    if soft_round is None:
      soft_round = self.soft_round[1]

    x = [tf.linspace(float(i[0]), float(i[1]), int(i[2])) for i in intervals]
    x = tf.meshgrid(*x, indexing="ij")
    x = tf.stack(x, axis=-1)

    y_hat, _, _ = self.encode_decode(x, False, False, soft_round, **kwargs)

    y = self.analysis(x)
    # We feed y here so we can visualize the full behavior of the synthesis
    # transform (not just at the quantized latent values).
    x_hat = self.synthesis(y)

    x = np.squeeze(x.numpy(), -1)
    y = np.squeeze(y.numpy(), -1)
    x_hat = np.squeeze(x_hat.numpy(), -1)
    y_hat = np.squeeze(y_hat.numpy(), -1)

    ylim = np.min(y), np.max(y)

    boundaries = np.nonzero(y_hat[1:] != y_hat[:-1])[0]
    lboundaries = (y_hat[boundaries] + y_hat[boundaries + 1]) / 2
    dboundaries = (x[boundaries] + x[boundaries + 1]) / 2

    lcodebook = np.unique(y_hat)
    dcodebook = self.synthesis(lcodebook[:, None]).numpy()
    dcodebook = np.squeeze(dcodebook, -1)
    mask = np.logical_and(ylim[0] < lcodebook, lcodebook < ylim[1])
    lcodebook = lcodebook[mask]
    dcodebook = dcodebook[mask]

    plt.figure(figsize=figsize or (16, 14))
    plt.plot(x, y, label="analysis transform")
    plt.plot(x_hat, y, label="synthesis transform")

    plt.gca().set_aspect("equal", "box")
    # Flip y axis if latent space is reversed.
    if y[0] > y[-1]:
      plt.gca().invert_yaxis()
    plt.xticks(dcodebook)
    plt.yticks(lcodebook)
    plt.grid(False)
    plt.xlabel("source space")
    plt.ylabel("latent space")

    xmin = plt.axis()[0]
    ymin = plt.axis()[2]
    for x, y in zip(dcodebook, lcodebook):
      plt.plot([xmin, x, x], [y, y, ymin], "black", lw=1)
      plt.plot(
          [x], [y], "black", marker="o", ms=5, lw=1,
          label="codebook" if x == dcodebook[0] else None)
    for x, y in zip(dboundaries, lboundaries):
      plt.plot([xmin, x, x], [y, y, ymin], "black", lw=1, ls=":")
      plt.plot(
          [x], [y], "black", marker="o", ms=3, lw=1, ls=":",
          label="boundaries" if x == dboundaries[0] else None)

    plt.legend(loc="upper left")

  def plot_jacobians(self, which, intervals, arrow_intervals, scale=2,
                     figsize=None):
    if not (len(intervals) == len(arrow_intervals) ==
            self.ndim_source == self.ndim_latent == 2):
      raise ValueError("This method is only defined for 2D models.")
    if which not in ("analysis", "synthesis"):
      raise ValueError("`which` must be 'analysis' or 'synthesis'.")

    data = [tf.linspace(float(i[0]), float(i[1]), int(i[2])) for i in intervals]
    data = tf.meshgrid(*data, indexing="ij")
    data = tf.stack(data, axis=-1)
    data_dist = self.source.prob(data).numpy()

    if which == "analysis":
      arrow_data = [
          tf.linspace(float(i[0]), float(i[1]), int(i[2]))
          for i in arrow_intervals
      ]
      arrow_data = tf.meshgrid(*arrow_data, indexing="ij")
      arrow_data = tf.stack(arrow_data, axis=-1)
      arrow_data = tf.reshape(arrow_data, (-1, arrow_data.shape[-1]))
      with tf.GradientTape(watch_accessed_variables=False) as tape:
        tape.watch(arrow_data)
        arrow_latents = self.analysis(arrow_data)
      # First dimension is batch, second is latent dim, third is source dim.
      jacobian = tape.batch_jacobian(arrow_latents, arrow_data)
      jacobian = tf.linalg.inv(jacobian)
      jacobian = tf.transpose(jacobian, (0, 2, 1))
    else:
      arrow_latents = [
          tf.linspace(float(i[0]), float(i[1]), int(i[2]))
          for i in arrow_intervals
      ]
      arrow_latents = tf.meshgrid(*arrow_latents, indexing="ij")
      arrow_latents = tf.stack(arrow_latents, axis=-1)
      arrow_latents = tf.reshape(arrow_latents, (-1, arrow_latents.shape[-1]))
      with tf.GradientTape(watch_accessed_variables=False) as tape:
        tape.watch(arrow_latents)
        arrow_data = self.synthesis(arrow_latents)
      jacobian = tape.batch_jacobian(arrow_data, arrow_latents)
      jacobian = tf.transpose(jacobian, (0, 2, 1))

    google_pink = (0xf4/255, 0x39/255, 0xa0/255)
    google_purple = (0xa1/255, 0x42/255, 0xf4/255)

    plt.figure(figsize=figsize or (16, 14))
    plt.imshow(
        data_dist, vmin=0, vmax=data_dist.max(), origin="lower",
        extent=(data[0, 0, 1], data[0, -1, 1], data[0, 0, 0], data[-1, 0, 0]))
    plt.quiver(
        arrow_data[:, 1], arrow_data[:, 0],
        jacobian[:, 0, 1], jacobian[:, 0, 0],
        pivot="tail", angles="xy", headlength=4, headaxislength=4, units="dots",
        color=google_pink, scale_units="xy", scale=scale,
    )
    plt.quiver(
        arrow_data[:, 1], arrow_data[:, 0],
        jacobian[:, 1, 1], jacobian[:, 1, 0],
        pivot="tail", angles="xy", headlength=4, headaxislength=4, units="dots",
        color=google_purple, scale_units="xy", scale=scale,
    )
    plt.axis("image")
    plt.grid(False)
    plt.xlim(data[0, 0, 1], data[0, -1, 1])
    plt.ylim(data[0, 0, 0], data[-1, 0, 0])
    plt.xlabel("source dimension 1")
    plt.ylabel("source dimension 2")

  def plot_prior_latent(self, intervals, figsize=None, **kwargs):
    if len(intervals) != self.ndim_latent or self.ndim_latent not in (1, 2):
      raise ValueError("This method is only defined for 1D or 2D models.")

    y = [tf.linspace(float(i[0]), float(i[1]), int(i[2])) for i in intervals]
    y = tf.meshgrid(*y, indexing="ij")
    y = tf.stack(y, axis=-1)
    prob = tfpd.Independent(self.prior(**kwargs), 1).prob(y)

    if self.ndim_latent == 1:
      plt.figure(figsize=figsize or (16, 6))
      plt.plot(y, prob)
      plt.xlabel("latent space")
      plt.ylabel("prior")
      plt.grid(True)
      plt.xticks(np.arange(*np.ceil(intervals[0][:2])))
    else:
      plt.figure(figsize=figsize or (16, 14))
      plt.imshow(
          prob, vmin=0, vmax=np.max(prob), origin="lower",
          extent=(y[0, 0, 1], y[0, -1, 1], y[0, 0, 0], y[-1, 0, 0]))
      plt.axis("image")
      plt.grid(False)
      plt.xlim(y[0, 0, 1], y[0, -1, 1])
      plt.ylim(y[0, 0, 0], y[-1, 0, 0])
      plt.xlabel("latent dimension 1")
      plt.ylabel("latent dimension 2")

  def plot_prior(self, figsize=None, xlim=None, **kwargs):
    prior = self.prior(**kwargs)
    assert not tuple(prior.event_shape)
    if not xlim:
      xlim = (tf.reduce_min(tfc.lower_tail(prior, 1e-5)),
              tf.reduce_max(tfc.upper_tail(prior, 1e-5)))
    y = tf.linspace(xlim[0], xlim[1], 1000)[:, None]
    prob = prior.prob(y)
    plt.figure(figsize=figsize or (16, 8))
    plt.plot(y, prob)
    plt.grid()
    plt.xlabel("latent space")
    plt.ylabel("$p$")
    plt.figure(figsize=figsize or (16, 8))
    plt.imshow(
        tf.transpose(prob), aspect="auto",
        extent=(xlim[0], xlim[1], -.5, self.ndim_latent - .5))
    plt.xlabel("latent space")
    plt.ylabel("latent dimension")
