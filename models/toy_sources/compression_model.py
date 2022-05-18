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
"""Base class for coding experiment."""

import abc
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


def source_dataset(source, batch_size, seed, dataset_size=None):
  """Returns a `tf.data.Dataset` of samples from `source`."""
  dataset = tf.data.Dataset.random(seed=seed)
  if dataset_size is not None:
    # This rounds up to multiple of batch size.
    batches = (dataset_size - 1) // batch_size + 1
    dataset = dataset.take(batches)
  return dataset.map(
      lambda seed: source.sample(batch_size, seed=tf.bitcast(seed, tf.int32)))


class CompressionModel(tf.keras.Model, metaclass=abc.ABCMeta):
  """Base class for coding experiment."""

  def __init__(self, source, lmbda, distortion_loss, **kwargs):
    super().__init__(**kwargs)
    self.source = source
    self.lmbda = float(lmbda)
    self.distortion_loss = str(distortion_loss)

  @property
  def ndim_source(self):
    return self.source.event_shape[0]

  @abc.abstractmethod
  def quantize(self, x):
    """Determines an equivalent vector quantizer for `x`.

    Args:
      x: A batch of source vectors.

    Returns:
      codebook: A codebook of vectors used to represent all elements of `x`.
      rates: For each codebook vector, the self-information in bits needed to
        encode it.
      indexes: Integer `Tensor`. For each batch element in `x`, returns the
        index into `codebook` that it is represented as.
    """

  @abc.abstractmethod
  def train_losses(self, x):
    """Computes the training losses for `x`.

    Args:
      x: A batch of source vectors.

    Returns:
      Either an RD loss value for each element in `x`, or a tuple which contains
      the rate and distortion losses for each element separately (as in
      `test_losses`).
    """

  @abc.abstractmethod
  def test_losses(self, x):
    """Computes the rate and distortion for each element of `x`.

    Args:
      x: A batch of source vectors.

    Returns:
      rates: For each element in `x`, the self-information in bits needed to
        encode it.
      distortions: The distortion loss for each element in `x`.
    """

  def distortion_fn(self, reference, reconstruction):
    reference = tf.cast(reference, self.dtype)
    if self.distortion_loss == "sse":
      diff = tf.math.squared_difference(reference, reconstruction)
      return tf.math.reduce_sum(diff, axis=-1)
    if self.distortion_loss == "mse":
      diff = tf.math.squared_difference(reference, reconstruction)
      return tf.math.reduce_mean(diff, axis=-1)

  def train_step(self, x):
    if hasattr(self, "alpha"):
      self.alpha = self.force_alpha
    with tf.GradientTape() as tape:
      rates, distortions = self.train_losses(x)
      losses = rates + self.lmbda * distortions
      loss = tf.math.reduce_mean(losses)
    variables = self.trainable_variables
    gradients = tape.gradient(loss, variables)
    self.optimizer.apply_gradients(zip(gradients, variables))
    self.loss.update_state(losses)
    self.rate.update_state(rates)
    self.distortion.update_state(distortions)
    energy = []
    size = []
    for grad in gradients:
      if grad is None:
        continue
      energy.append(tf.reduce_sum(tf.square(tf.cast(grad, tf.float64))))
      size.append(tf.cast(tf.size(grad), tf.float64))
    self.grad_rms.update_state(tf.sqrt(tf.add_n(energy) / tf.add_n(size)))
    return {
        m.name: m.result()
        for m in [self.loss, self.rate, self.distortion, self.grad_rms]
    }

  def test_step(self, x):
    rates, distortions = self.test_losses(x)
    losses = rates + self.lmbda * distortions
    self.loss.update_state(losses)
    self.rate.update_state(rates)
    self.distortion.update_state(distortions)
    return {m.name: m.result() for m in [self.loss, self.rate, self.distortion]}

  def compile(self, **kwargs):
    super().compile(
        loss=None,
        metrics=None,
        loss_weights=None,
        weighted_metrics=None,
        **kwargs,
    )
    self.loss = tf.keras.metrics.Mean(name="loss")
    self.rate = tf.keras.metrics.Mean(name="rate")
    self.distortion = tf.keras.metrics.Mean(name="distortion")
    self.grad_rms = tf.keras.metrics.Mean(name="gradient RMS")

  def fit(self, batch_size, validation_size, validation_batch_size,
          train_size=None, train_seed=None, validation_seed=82913749, **kwargs):
    train_data = source_dataset(
        self.source, batch_size, train_seed, dataset_size=train_size)
    if train_size is not None:
      train_data = train_data.repeat()
    validation_data = source_dataset(
        self.source, validation_batch_size, validation_seed,
        dataset_size=validation_size)
    super().fit(
        train_data,
        validation_data=validation_data,
        shuffle=False,
        **kwargs,
    )

  def plot_quantization(self, intervals, figsize=None, **kwargs):
    if len(intervals) != self.ndim_source or self.ndim_source not in (1, 2):
      raise ValueError("This method is only defined for 1D or 2D models.")

    data = [tf.linspace(float(i[0]), float(i[1]), int(i[2])) for i in intervals]
    data = tf.meshgrid(*data, indexing="ij")
    data = tf.stack(data, axis=-1)

    codebook, rates, indexes = self.quantize(data, **kwargs)
    codebook = codebook.numpy()
    rates = rates.numpy()
    indexes = indexes.numpy()

    data_dist = self.source.prob(data).numpy()
    counts = np.bincount(np.ravel(indexes), minlength=len(codebook))
    prior = 2 ** (-rates)

    if self.ndim_source == 1:
      data = np.squeeze(data, axis=-1)
      boundaries = np.nonzero(indexes[1:] != indexes[:-1])[0]
      boundaries = (data[boundaries] + data[boundaries + 1]) / 2
      plt.figure(figsize=figsize or (16, 8))
      plt.plot(data, data_dist, label="source")
      markers, stems, base = plt.stem(
          codebook[counts > 0], prior[counts > 0], label="codebook")
      plt.setp(markers, color="black")
      plt.setp(stems, color="black")
      plt.setp(base, linestyle="None")
      plt.xticks(np.sort(codebook[counts > 0]))
      plt.grid(False, axis="x")
      for r in boundaries:
        plt.axvline(
            r, color="black", lw=1, ls=":",
            label="boundaries" if r == boundaries[0] else None)
      plt.xlim(np.min(data), np.max(data))
      plt.ylim(bottom=-.01)
      plt.legend(loc="upper left")
      plt.xlabel("source space")
    else:
      google_pink = (0xf4/255, 0x39/255, 0xa0/255)
      plt.figure(figsize=figsize or (16, 14))
      vmax = data_dist.max()
      plt.imshow(
          data_dist, vmin=0, vmax=vmax, origin="lower",
          extent=(
              data[0, 0, 1], data[0, -1, 1], data[0, 0, 0], data[-1, 0, 0]))
      plt.contour(
          data[:, :, 1], data[:, :, 0], indexes,
          np.arange(len(codebook)) + .5,
          colors=[google_pink], linewidths=.5)
      plt.plot(
          codebook[counts > 0, 1], codebook[counts > 0, 0],
          "o", color=google_pink)
      plt.axis("image")
      plt.grid(False)
      plt.xlim(data[0, 0, 1], data[0, -1, 1])
      plt.ylim(data[0, 0, 0], data[-1, 0, 0])
      plt.xlabel("source dimension 1")
      plt.ylabel("source dimension 2")
