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
"""Variational entropy-constrained vector quantization."""

import tensorflow as tf
from toy_sources import compression_model


class VECVQModel(compression_model.CompressionModel):
  """Variational entropy-constrained vector quantization model."""

  def __init__(self, codebook_size, initialize="sample", logit_scale=1e0,
               **kwargs):
    super().__init__(**kwargs)
    self.codebook_size = int(codebook_size)
    self.logit_scale = float(logit_scale)

    # Initialize codebook with samples from the source distribution.
    if initialize.startswith("sample"):
      codebook_init = self.source.sample(self.codebook_size)
      if len(initialize) > 6:
        scale = float(initialize[7:])
        codebook_init += tf.random.normal(codebook_init.shape, stddev=scale)
    elif initialize.startswith("uniform-"):
      width = float(initialize[8:])
      codebook_init = tf.random.uniform(
          [self.codebook_size, self.ndim_source],
          minval=-width/2, maxval=width/2)
    assert codebook_init.shape == (self.codebook_size, self.ndim_source)
    logits_init = tf.random.normal(
        [self.codebook_size], stddev=self.logit_scale/10)
    self.codebook = tf.Variable(
        tf.cast(codebook_init, self.dtype), name="codebook")
    self._logits = tf.Variable(
        tf.cast(logits_init, self.dtype), name="logits")

  @property
  def logits(self):
    return self._logits / self.logit_scale

  def all_rd(self, x):
    rates = tf.math.reduce_logsumexp(self.logits) - self.logits
    rates /= tf.cast(tf.math.log(2.), dtype=self.dtype)
    distortions = self.distortion_fn(tf.expand_dims(x, -2), self.codebook)
    return rates, distortions

  def quantize(self, x):
    rates, distortions = self.all_rd(x)
    all_rd = rates + self.lmbda * distortions
    indexes = tf.argmin(all_rd, axis=-1, output_type=tf.int32)
    return self.codebook, rates, indexes

  def test_losses(self, x):
    rates, distortions = self.all_rd(x)
    all_rd = rates + self.lmbda * distortions
    indexes = tf.argmin(all_rd, axis=-1, output_type=tf.int32)
    rates = tf.gather(rates, indexes)
    distortions = tf.gather(distortions, indexes, batch_dims=indexes.shape.rank)
    return rates, distortions

  train_losses = test_losses
