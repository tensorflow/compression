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
"""Nonlinear transform coder with hyperprior for RGB images.

This is the image compression model published in:
D. Minnen and S. Singh:
"Channel-wise autoregressive entropy models for learned image compression"
Int. Conf. on Image Compression (ICIP), 2020
https://arxiv.org/abs/2007.08739

This is meant as 'educational' code - you can use this to get started with your
own experiments. To reproduce the exact results from the paper, tuning of hyper-
parameters may be necessary. To compress images with published models, see
`tfci.py`.

This script requires TFC v2 (`pip install tensorflow-compression==2.*`).
"""

import argparse
import functools
import glob
import sys
from absl import app
from absl.flags import argparse_flags
import tensorflow as tf
import tensorflow_compression as tfc
import tensorflow_datasets as tfds


def read_png(filename):
  """Loads a PNG image file."""
  string = tf.io.read_file(filename)
  return tf.image.decode_image(string, channels=3)


def write_png(filename, image):
  """Saves an image to a PNG file."""
  string = tf.image.encode_png(image)
  tf.io.write_file(filename, string)


class AnalysisTransform(tf.keras.Sequential):
  """The analysis transform."""

  def __init__(self, latent_depth):
    super().__init__()
    conv = functools.partial(tfc.SignalConv2D, corr=True, strides_down=2,
                             padding="same_zeros", use_bias=True)
    layers = [
        tf.keras.layers.Lambda(lambda x: x / 255.),
        conv(192, (5, 5), name="layer_0", activation=tfc.GDN(name="gdn_0")),
        conv(192, (5, 5), name="layer_1", activation=tfc.GDN(name="gdn_1")),
        conv(192, (5, 5), name="layer_2", activation=tfc.GDN(name="gdn_2")),
        conv(latent_depth, (5, 5), name="layer_3", activation=None),
    ]
    for layer in layers:
      self.add(layer)


class SynthesisTransform(tf.keras.Sequential):
  """The synthesis transform."""

  def __init__(self):
    super().__init__()
    conv = functools.partial(tfc.SignalConv2D, corr=False, strides_up=2,
                             padding="same_zeros", use_bias=True)
    layers = [
        conv(192, (5, 5), name="layer_0",
             activation=tfc.GDN(name="igdn_0", inverse=True)),
        conv(192, (5, 5), name="layer_1",
             activation=tfc.GDN(name="igdn_1", inverse=True)),
        conv(192, (5, 5), name="layer_2",
             activation=tfc.GDN(name="igdn_2", inverse=True)),
        conv(3, (5, 5), name="layer_3",
             activation=None),
        tf.keras.layers.Lambda(lambda x: x * 255.),
    ]
    for layer in layers:
      self.add(layer)


class HyperAnalysisTransform(tf.keras.Sequential):
  """The analysis transform for the entropy model parameters."""

  def __init__(self, hyperprior_depth):
    super().__init__()
    conv = functools.partial(tfc.SignalConv2D, corr=True, padding="same_zeros")

    # See Appendix C.2 for more information on using a small hyperprior.
    layers = [
        conv(320, (3, 3), name="layer_0", strides_down=1, use_bias=True,
             activation=tf.nn.relu),
        conv(256, (5, 5), name="layer_1", strides_down=2, use_bias=True,
             activation=tf.nn.relu),
        conv(hyperprior_depth, (5, 5), name="layer_2", strides_down=2,
             use_bias=False, activation=None),
    ]
    for layer in layers:
      self.add(layer)


class HyperSynthesisTransform(tf.keras.Sequential):
  """The synthesis transform for the entropy model parameters."""

  def __init__(self):
    super().__init__()
    conv = functools.partial(
        tfc.SignalConv2D, corr=False, padding="same_zeros", use_bias=True,
        kernel_parameter="variable", activation=tf.nn.relu)

    # Note that the output tensor is still latent (it represents means and
    # scales but it does NOT hold mean or scale values explicitly). Therefore,
    # the final activation is ReLU rather than None or Exp). For the same
    # reason, it is not a requirement that the final depth of this transform
    # matches the depth of `y`.
    layers = [
        conv(192, (5, 5), name="layer_0", strides_up=2),
        conv(256, (5, 5), name="layer_1", strides_up=2),
        conv(320, (3, 3), name="layer_2", strides_up=1),
    ]
    for layer in layers:
      self.add(layer)


class SliceTransform(tf.keras.layers.Layer):
  """Transform for channel-conditional params and latent residual prediction."""

  def __init__(self, latent_depth, num_slices):
    super().__init__()
    conv = functools.partial(
        tfc.SignalConv2D, corr=False, strides_up=1, padding="same_zeros",
        use_bias=True, kernel_parameter="variable")

    # Note that the number of channels in the output tensor must match the
    # size of the corresponding slice. If we have 10 slices and a bottleneck
    # with 320 channels, the output is 320 / 10 = 32 channels.
    slice_depth = latent_depth // num_slices
    if slice_depth * num_slices != latent_depth:
      raise ValueError("Slices do not evenly divide latent depth (%d / %d)" % (
          latent_depth, num_slices))

    self.transform = tf.keras.Sequential([
        conv(224, (5, 5), name="layer_0", activation=tf.nn.relu),
        conv(128, (5, 5), name="layer_1", activation=tf.nn.relu),
        conv(slice_depth, (3, 3), name="layer_2", activation=None),
    ])

  def call(self, tensor):
    return self.transform(tensor)


class MS2020Model(tf.keras.Model):
  """Main model class."""

  def __init__(self, lmbda,
               num_filters, latent_depth, hyperprior_depth,
               num_slices, max_support_slices,
               num_scales, scale_min, scale_max):
    super().__init__()
    self.lmbda = lmbda
    self.num_scales = num_scales
    self.num_slices = num_slices
    self.max_support_slices = max_support_slices
    offset = tf.math.log(scale_min)
    factor = (tf.math.log(scale_max) - tf.math.log(scale_min)) / (
        num_scales - 1.)
    self.scale_fn = lambda i: tf.math.exp(offset + factor * i)
    self.analysis_transform = AnalysisTransform(latent_depth)
    self.synthesis_transform = SynthesisTransform()
    self.hyper_analysis_transform = HyperAnalysisTransform(hyperprior_depth)
    self.hyper_synthesis_mean_transform = HyperSynthesisTransform()
    self.hyper_synthesis_scale_transform = HyperSynthesisTransform()
    self.cc_mean_transforms = [
        SliceTransform(latent_depth, num_slices) for _ in range(num_slices)]
    self.cc_scale_transforms = [
        SliceTransform(latent_depth, num_slices) for _ in range(num_slices)]
    self.lrp_transforms = [
        SliceTransform(latent_depth, num_slices) for _ in range(num_slices)]
    self.hyperprior = tfc.NoisyDeepFactorized(batch_shape=[hyperprior_depth])
    self.build((None, None, None, 3))
    # The call signature of decompress() depends on the number of slices, so we
    # need to compile the function dynamically.
    self.decompress = tf.function(
        input_signature=3 * [tf.TensorSpec(shape=(2,), dtype=tf.int32)] +
        (num_slices + 1) * [tf.TensorSpec(shape=(1,), dtype=tf.string)]
    )(self.decompress)

  def call(self, x, training):
    """Computes rate and distortion losses."""
    x = tf.cast(x, self.compute_dtype)  # TODO(jonycgn): Why is this necessary?
    # Build the encoder (analysis) half of the hierarchical autoencoder.
    y = self.analysis_transform(x)
    y_shape = tf.shape(y)[1:-1]

    z = self.hyper_analysis_transform(y)

    num_pixels = tf.cast(tf.reduce_prod(tf.shape(x)[1:-1]), tf.float32)

    # Build the entropy model for the hyperprior (z).
    em_z = tfc.ContinuousBatchedEntropyModel(
        self.hyperprior, coding_rank=3, compression=False,
        offset_heuristic=False)

    # When training, z_bpp is based on the noisy version of z (z_tilde).
    _, z_bits = em_z(z, training=training)
    z_bpp = tf.reduce_mean(z_bits) / num_pixels

    # Use rounding (instead of uniform noise) to modify z before passing it
    # to the hyper-synthesis transforms. Note that quantize() overrides the
    # gradient to create a straight-through estimator.
    z_hat = em_z.quantize(z)

    # Build the decoder (synthesis) half of the hierarchical autoencoder.
    latent_scales = self.hyper_synthesis_scale_transform(z_hat)
    latent_means = self.hyper_synthesis_mean_transform(z_hat)

    # Build a conditional entropy model for the slices.
    em_y = tfc.LocationScaleIndexedEntropyModel(
        tfc.NoisyNormal, num_scales=self.num_scales, scale_fn=self.scale_fn,
        coding_rank=3, compression=False)

    # En/Decode each slice conditioned on hyperprior and previous slices.
    y_slices = tf.split(y, self.num_slices, axis=-1)
    y_hat_slices = []
    y_bpps = []
    for slice_index, y_slice in enumerate(y_slices):
      # Model may condition on only a subset of previous slices.
      support_slices = (y_hat_slices if self.max_support_slices < 0 else
                        y_hat_slices[:self.max_support_slices])

      # Predict mu and sigma for the current slice.
      mean_support = tf.concat([latent_means] + support_slices, axis=-1)
      mu = self.cc_mean_transforms[slice_index](mean_support)
      mu = mu[:, :y_shape[0], :y_shape[1], :]

      # Note that in this implementation, `sigma` represents scale indices,
      # not actual scale values.
      scale_support = tf.concat([latent_scales] + support_slices, axis=-1)
      sigma = self.cc_scale_transforms[slice_index](scale_support)
      sigma = sigma[:, :y_shape[0], :y_shape[1], :]

      _, slice_bits = em_y(y_slice, sigma, loc=mu, training=training)
      slice_bpp = tf.reduce_mean(slice_bits) / num_pixels
      y_bpps.append(slice_bpp)

      # For the synthesis transform, use rounding. Note that quantize()
      # overrides the gradient to create a straight-through estimator.
      y_hat_slice = em_y.quantize(y_slice, loc=mu)

      # Add latent residual prediction (LRP).
      lrp_support = tf.concat([mean_support, y_hat_slice], axis=-1)
      lrp = self.lrp_transforms[slice_index](lrp_support)
      lrp = 0.5 * tf.math.tanh(lrp)
      y_hat_slice += lrp

      y_hat_slices.append(y_hat_slice)

    # Merge slices and generate the image reconstruction.
    y_hat = tf.concat(y_hat_slices, axis=-1)
    x_hat = self.synthesis_transform(y_hat)

    # Total bpp is sum of bpp from hyperprior and all slices.
    total_bpp = tf.add_n(y_bpps + [z_bpp])

    # Mean squared error across pixels.
    # Don't clip or round pixel values while training.
    mse = tf.reduce_mean(tf.math.squared_difference(x, x_hat))
    mse = tf.cast(mse, total_bpp.dtype)

    # Calculate and return the rate-distortion loss: R + lambda * D.
    loss = total_bpp + self.lmbda * mse

    return loss, total_bpp, mse

  def train_step(self, x):
    with tf.GradientTape() as tape:
      loss, bpp, mse = self(x, training=True)
    variables = self.trainable_variables
    gradients = tape.gradient(loss, variables)
    self.optimizer.apply_gradients(zip(gradients, variables))
    self.loss.update_state(loss)
    self.bpp.update_state(bpp)
    self.mse.update_state(mse)
    return {m.name: m.result() for m in [self.loss, self.bpp, self.mse]}

  def test_step(self, x):
    loss, bpp, mse = self(x, training=False)
    self.loss.update_state(loss)
    self.bpp.update_state(bpp)
    self.mse.update_state(mse)
    return {m.name: m.result() for m in [self.loss, self.bpp, self.mse]}

  def predict_step(self, x):
    raise NotImplementedError("Prediction API is not supported.")

  def compile(self, **kwargs):
    super().compile(
        loss=None,
        metrics=None,
        loss_weights=None,
        weighted_metrics=None,
        **kwargs,
    )
    self.loss = tf.keras.metrics.Mean(name="loss")
    self.bpp = tf.keras.metrics.Mean(name="bpp")
    self.mse = tf.keras.metrics.Mean(name="mse")

  def fit(self, *args, **kwargs):
    retval = super().fit(*args, **kwargs)
    # After training, fix range coding tables.
    self.em_z = tfc.ContinuousBatchedEntropyModel(
        self.hyperprior, coding_rank=3, compression=True,
        offset_heuristic=False)
    self.em_y = tfc.LocationScaleIndexedEntropyModel(
        tfc.NoisyNormal, num_scales=self.num_scales, scale_fn=self.scale_fn,
        coding_rank=3, compression=True)
    return retval

  @tf.function(input_signature=[
      tf.TensorSpec(shape=(None, None, 3), dtype=tf.uint8),
  ])
  def compress(self, x):
    """Compresses an image."""
    # Add batch dimension and cast to float.
    x = tf.expand_dims(x, 0)
    x = tf.cast(x, dtype=self.compute_dtype)

    y_strings = []
    x_shape = tf.shape(x)[1:-1]

    # Build the encoder (analysis) half of the hierarchical autoencoder.
    y = self.analysis_transform(x)
    y_shape = tf.shape(y)[1:-1]

    z = self.hyper_analysis_transform(y)
    z_shape = tf.shape(z)[1:-1]

    z_string = self.em_z.compress(z)
    z_hat = self.em_z.decompress(z_string, z_shape)

    # Build the decoder (synthesis) half of the hierarchical autoencoder.
    latent_scales = self.hyper_synthesis_scale_transform(z_hat)
    latent_means = self.hyper_synthesis_mean_transform(z_hat)

    # En/Decode each slice conditioned on hyperprior and previous slices.
    y_slices = tf.split(y, self.num_slices, axis=-1)
    y_hat_slices = []
    for slice_index, y_slice in enumerate(y_slices):
      # Model may condition on only a subset of previous slices.
      support_slices = (y_hat_slices if self.max_support_slices < 0 else
                        y_hat_slices[:self.max_support_slices])

      # Predict mu and sigma for the current slice.
      mean_support = tf.concat([latent_means] + support_slices, axis=-1)
      mu = self.cc_mean_transforms[slice_index](mean_support)
      mu = mu[:, :y_shape[0], :y_shape[1], :]

      # Note that in this implementation, `sigma` represents scale indices,
      # not actual scale values.
      scale_support = tf.concat([latent_scales] + support_slices, axis=-1)
      sigma = self.cc_scale_transforms[slice_index](scale_support)
      sigma = sigma[:, :y_shape[0], :y_shape[1], :]

      slice_string = self.em_y.compress(y_slice, sigma, mu)
      y_strings.append(slice_string)
      y_hat_slice = self.em_y.decompress(slice_string, sigma, mu)

      # Add latent residual prediction (LRP).
      lrp_support = tf.concat([mean_support, y_hat_slice], axis=-1)
      lrp = self.lrp_transforms[slice_index](lrp_support)
      lrp = 0.5 * tf.math.tanh(lrp)
      y_hat_slice += lrp

      y_hat_slices.append(y_hat_slice)

    return (x_shape, y_shape, z_shape, z_string) + tuple(y_strings)

  def decompress(self, x_shape, y_shape, z_shape, z_string, *y_strings):
    """Decompresses an image."""
    assert len(y_strings) == self.num_slices

    z_hat = self.em_z.decompress(z_string, z_shape)

    # Build the decoder (synthesis) half of the hierarchical autoencoder.
    latent_scales = self.hyper_synthesis_scale_transform(z_hat)
    latent_means = self.hyper_synthesis_mean_transform(z_hat)

    # En/Decode each slice conditioned on hyperprior and previous slices.
    y_hat_slices = []
    for slice_index, y_string in enumerate(y_strings):
      # Model may condition on only a subset of previous slices.
      support_slices = (y_hat_slices if self.max_support_slices < 0 else
                        y_hat_slices[:self.max_support_slices])

      # Predict mu and sigma for the current slice.
      mean_support = tf.concat([latent_means] + support_slices, axis=-1)
      mu = self.cc_mean_transforms[slice_index](mean_support)
      mu = mu[:, :y_shape[0], :y_shape[1], :]

      # Note that in this implementation, `sigma` represents scale indices,
      # not actual scale values.
      scale_support = tf.concat([latent_scales] + support_slices, axis=-1)
      sigma = self.cc_scale_transforms[slice_index](scale_support)
      sigma = sigma[:, :y_shape[0], :y_shape[1], :]

      y_hat_slice = self.em_y.decompress(y_string, sigma, loc=mu)

      # Add latent residual prediction (LRP).
      lrp_support = tf.concat([mean_support, y_hat_slice], axis=-1)
      lrp = self.lrp_transforms[slice_index](lrp_support)
      lrp = 0.5 * tf.math.tanh(lrp)
      y_hat_slice += lrp

      y_hat_slices.append(y_hat_slice)

    # Merge slices and generate the image reconstruction.
    y_hat = tf.concat(y_hat_slices, axis=-1)
    x_hat = self.synthesis_transform(y_hat)
    # Remove batch dimension, and crop away any extraneous padding.
    x_hat = x_hat[0, :x_shape[0], :x_shape[1], :]
    # Then cast back to 8-bit integer.
    return tf.saturate_cast(tf.round(x_hat), tf.uint8)


def check_image_size(image, patchsize):
  shape = tf.shape(image)
  return shape[0] >= patchsize and shape[1] >= patchsize and shape[-1] == 3


def crop_image(image, patchsize):
  image = tf.image.random_crop(image, (patchsize, patchsize, 3))
  return tf.cast(image, tf.keras.mixed_precision.global_policy().compute_dtype)


def get_dataset(name, split, args):
  """Creates input data pipeline from a TF Datasets dataset."""
  with tf.device("/cpu:0"):
    dataset = tfds.load(name, split=split, shuffle_files=True)
    if split == "train":
      dataset = dataset.repeat()
    dataset = dataset.filter(
        lambda x: check_image_size(x["image"], args.patchsize))
    dataset = dataset.map(
        lambda x: crop_image(x["image"], args.patchsize))
    dataset = dataset.batch(args.batchsize, drop_remainder=True)
  return dataset


def get_custom_dataset(split, args):
  """Creates input data pipeline from custom PNG images."""
  with tf.device("/cpu:0"):
    files = glob.glob(args.train_glob)
    if not files:
      raise RuntimeError(f"No training images found with glob "
                         f"'{args.train_glob}'.")
    dataset = tf.data.Dataset.from_tensor_slices(files)
    dataset = dataset.shuffle(len(files), reshuffle_each_iteration=True)
    if split == "train":
      dataset = dataset.repeat()
    dataset = dataset.map(
        lambda x: crop_image(read_png(x), args.patchsize),
        num_parallel_calls=args.preprocess_threads)
    dataset = dataset.batch(args.batchsize, drop_remainder=True)
  return dataset


def train(args):
  """Instantiates and trains the model."""
  if args.precision_policy:
    tf.keras.mixed_precision.set_global_policy(args.precision_policy)
  if args.check_numerics:
    tf.debugging.enable_check_numerics()

  model = MS2020Model(
      args.lmbda, args.num_filters, args.latent_depth, args.hyperprior_depth,
      args.num_slices, args.max_support_slices,
      args.num_scales, args.scale_min, args.scale_max)
  model.compile(
      optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
  )

  if args.train_glob:
    train_dataset = get_custom_dataset("train", args)
    validation_dataset = get_custom_dataset("validation", args)
  else:
    train_dataset = get_dataset("clic", "train", args)
    validation_dataset = get_dataset("clic", "validation", args)
  validation_dataset = validation_dataset.take(args.max_validation_steps)

  model.fit(
      train_dataset.prefetch(8),
      epochs=args.epochs,
      steps_per_epoch=args.steps_per_epoch,
      validation_data=validation_dataset.cache(),
      validation_freq=1,
      callbacks=[
          tf.keras.callbacks.TerminateOnNaN(),
          tf.keras.callbacks.TensorBoard(
              log_dir=args.train_path,
              histogram_freq=1, update_freq="epoch"),
          tf.keras.callbacks.BackupAndRestore(args.train_path),
      ],
      verbose=int(args.verbose),
  )
  model.save(args.model_path)


def compress(args):
  """Compresses an image."""
  # Load model and use it to compress the image.
  model = tf.keras.models.load_model(args.model_path)
  x = read_png(args.input_file)
  tensors = model.compress(x)

  # Write a binary file with the shape information and the compressed string.
  packed = tfc.PackedTensors()
  packed.pack(tensors)
  with open(args.output_file, "wb") as f:
    f.write(packed.string)

  # If requested, decompress the image and measure performance.
  if args.verbose:
    x_hat = model.decompress(*tensors)

    # Cast to float in order to compute metrics.
    x = tf.cast(x, tf.float32)
    x_hat = tf.cast(x_hat, tf.float32)
    mse = tf.reduce_mean(tf.math.squared_difference(x, x_hat))
    psnr = tf.squeeze(tf.image.psnr(x, x_hat, 255))
    msssim = tf.squeeze(tf.image.ssim_multiscale(x, x_hat, 255))
    msssim_db = -10. * tf.math.log(1 - msssim) / tf.math.log(10.)

    # The actual bits per pixel including entropy coding overhead.
    num_pixels = tf.reduce_prod(tf.shape(x)[:-1])
    bpp = len(packed.string) * 8 / num_pixels

    print(f"Mean squared error: {mse:0.4f}")
    print(f"PSNR (dB): {psnr:0.2f}")
    print(f"Multiscale SSIM: {msssim:0.4f}")
    print(f"Multiscale SSIM (dB): {msssim_db:0.2f}")
    print(f"Bits per pixel: {bpp:0.4f}")


def decompress(args):
  """Decompresses an image."""
  # Load the model and determine the dtypes of tensors required to decompress.
  model = tf.keras.models.load_model(args.model_path)
  dtypes = [t.dtype for t in model.decompress.input_signature]

  # Read the shape information and compressed string from the binary file,
  # and decompress the image using the model.
  with open(args.input_file, "rb") as f:
    packed = tfc.PackedTensors(f.read())
  tensors = packed.unpack(dtypes)
  x_hat = model.decompress(*tensors)

  # Write reconstructed image out as a PNG file.
  write_png(args.output_file, x_hat)


def parse_args(argv):
  """Parses command line arguments."""
  parser = argparse_flags.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)

  # High-level options.
  parser.add_argument(
      "--verbose", "-V", action="store_true",
      help="Report progress and metrics when training or compressing.")
  parser.add_argument(
      "--model_path", default="ms2020",
      help="Path where to save/load the trained model.")
  subparsers = parser.add_subparsers(
      title="commands", dest="command",
      help="What to do: 'train' loads training data and trains (or continues "
           "to train) a new model. 'compress' reads an image file (lossless "
           "PNG format) and writes a compressed binary file. 'decompress' "
           "reads a binary file and reconstructs the image (in PNG format). "
           "input and output filenames need to be provided for the latter "
           "two options. Invoke '<command> -h' for more information.")

  # 'train' subcommand.
  train_cmd = subparsers.add_parser(
      "train",
      formatter_class=argparse.ArgumentDefaultsHelpFormatter,
      description="Trains (or continues to train) a new model. Note that this "
                  "model trains on a continuous stream of patches drawn from "
                  "the training image dataset. An epoch is always defined as "
                  "the same number of batches given by --steps_per_epoch. "
                  "The purpose of validation is mostly to evaluate the "
                  "rate-distortion performance of the model using actual "
                  "quantization rather than the differentiable proxy loss. "
                  "Note that when using custom training images, the validation "
                  "set is simply a random sampling of patches from the "
                  "training set.")
  train_cmd.add_argument(
      "--lambda", type=float, default=0.01, dest="lmbda",
      help="Lambda for rate-distortion tradeoff.")
  train_cmd.add_argument(
      "--train_glob", type=str, default=None,
      help="Glob pattern identifying custom training data. This pattern must "
           "expand to a list of RGB images in PNG format. If unspecified, the "
           "CLIC dataset from TensorFlow Datasets is used.")
  train_cmd.add_argument(
      "--num_filters", type=int, default=192,
      help="Number of filters per layer.")
  train_cmd.add_argument(
      "--latent_depth", type=int, default=320,
      help="Number of filters of the last layer of the analysis transform.")
  train_cmd.add_argument(
      "--hyperprior_depth", type=int, default=192,
      help="Number of filters of the last layer of the hyper-analysis "
           "transform.")
  train_cmd.add_argument(
      "--num_slices", type=int, default=10,
      help="Number of channel slices for conditional entropy modeling.")
  train_cmd.add_argument(
      "--max_support_slices", type=int, default=5,
      help="Maximum number of preceding slices to condition the current slice "
           "on. See Appendix C.1 of the paper for details.")
  train_cmd.add_argument(
      "--num_scales", type=int, default=64,
      help="Number of Gaussian scales to prepare range coding tables for.")
  train_cmd.add_argument(
      "--scale_min", type=float, default=.11,
      help="Minimum value of standard deviation of Gaussians.")
  train_cmd.add_argument(
      "--scale_max", type=float, default=256.,
      help="Maximum value of standard deviation of Gaussians.")
  train_cmd.add_argument(
      "--train_path", default="/tmp/train_ms2020",
      help="Path where to log training metrics for TensorBoard and back up "
           "intermediate model checkpoints.")
  train_cmd.add_argument(
      "--batchsize", type=int, default=8,
      help="Batch size for training and validation.")
  train_cmd.add_argument(
      "--patchsize", type=int, default=256,
      help="Size of image patches for training and validation.")
  train_cmd.add_argument(
      "--epochs", type=int, default=1000,
      help="Train up to this number of epochs. (One epoch is here defined as "
           "the number of steps given by --steps_per_epoch, not iterations "
           "over the full training dataset.)")
  train_cmd.add_argument(
      "--steps_per_epoch", type=int, default=1000,
      help="Perform validation and produce logs after this many batches.")
  train_cmd.add_argument(
      "--max_validation_steps", type=int, default=16,
      help="Maximum number of batches to use for validation. If -1, use one "
           "patch from each image in the training set.")
  train_cmd.add_argument(
      "--preprocess_threads", type=int, default=16,
      help="Number of CPU threads to use for parallel decoding of training "
           "images.")
  train_cmd.add_argument(
      "--precision_policy", type=str, default=None,
      help="Policy for `tf.keras.mixed_precision` training.")
  train_cmd.add_argument(
      "--check_numerics", action="store_true",
      help="Enable TF support for catching NaN and Inf in tensors.")

  # 'compress' subcommand.
  compress_cmd = subparsers.add_parser(
      "compress",
      formatter_class=argparse.ArgumentDefaultsHelpFormatter,
      description="Reads a PNG file, compresses it, and writes a TFCI file.")

  # 'decompress' subcommand.
  decompress_cmd = subparsers.add_parser(
      "decompress",
      formatter_class=argparse.ArgumentDefaultsHelpFormatter,
      description="Reads a TFCI file, reconstructs the image, and writes back "
                  "a PNG file.")

  # Arguments for both 'compress' and 'decompress'.
  for cmd, ext in ((compress_cmd, ".tfci"), (decompress_cmd, ".png")):
    cmd.add_argument(
        "input_file",
        help="Input filename.")
    cmd.add_argument(
        "output_file", nargs="?",
        help=f"Output filename (optional). If not provided, appends '{ext}' to "
             f"the input filename.")

  # Parse arguments.
  args = parser.parse_args(argv[1:])
  if args.command is None:
    parser.print_usage()
    sys.exit(2)
  return args


def main(args):
  # Invoke subcommand.
  if args.command == "train":
    train(args)
  elif args.command == "compress":
    if not args.output_file:
      args.output_file = args.input_file + ".tfci"
    compress(args)
  elif args.command == "decompress":
    if not args.output_file:
      args.output_file = args.input_file + ".png"
    decompress(args)


if __name__ == "__main__":
  app.run(main, flags_parser=parse_args)
