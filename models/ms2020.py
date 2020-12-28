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

This script requires tensorflow-compression v2.x.
"""

import argparse
import functools
import glob
import sys

from absl import app
from absl.flags import argparse_flags
import numpy as np
import tensorflow as tf

import tensorflow_compression as tfc

LATENT_DEPTH = 320
HYPERPRIOR_DEPTH = 192
NUM_SLICES = 10
MAX_SUPPORT_SLICES = 5  # see Appendix C.1 for details on partial support

SCALES_MIN = 0.11
SCALES_MAX = 256
SCALES_LEVELS = 64


def quantize_image(image):
  """Convert [0..1] float image to [0..255] uint8."""
  image = tf.round(image * 255)
  image = tf.saturate_cast(image, tf.uint8)
  return image


def read_png(filename):
  """Loads and returns a PNG image file with values in [0..1]."""
  string = tf.io.read_file(filename)
  image = tf.image.decode_image(string, channels=3)
  image = tf.cast(image, tf.float32)
  image /= 255
  return image


def write_png(filename, image):
  """Saves an image with [0..1] pixel values to a PNG file."""
  image = quantize_image(image)
  string = tf.image.encode_png(image)
  return tf.io.write_file(filename, string)


def scale_fn(indexes):
  """Calculates scale values from scale indices."""
  a = tf.constant(np.log(SCALES_MIN), dtype=tf.float32)
  b = tf.constant((np.log(SCALES_MAX) - np.log(SCALES_MIN)) /
                  (SCALES_LEVELS - 1), dtype=tf.float32)
  return tf.math.exp(a + b * tf.cast(indexes, tf.float32))


class AnalysisTransform(tf.keras.Sequential):
  """The analysis transform."""

  def __init__(self):
    super().__init__()
    conv = functools.partial(tfc.SignalConv2D, corr=True, strides_down=2,
                             padding="same_zeros", use_bias=True)
    layers = [
        conv(192, (5, 5), name="layer_0", activation=tfc.GDN(name="gdn_0")),
        conv(192, (5, 5), name="layer_1", activation=tfc.GDN(name="gdn_1")),
        conv(192, (5, 5), name="layer_2", activation=tfc.GDN(name="gdn_2")),
        conv(LATENT_DEPTH, (5, 5), name="layer_3", activation=None),
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
    ]
    for layer in layers:
      self.add(layer)


class HyperAnalysisTransform(tf.keras.Sequential):
  """The analysis transform for the entropy model parameters."""

  def __init__(self):
    super().__init__()
    conv = functools.partial(tfc.SignalConv2D, corr=True, padding="same_zeros")

    # See Appendix C.2 for more information on using a small hyperprior.
    layers = [
        conv(320, (3, 3), name="layer_0", strides_down=1, use_bias=True,
             activation=tf.nn.relu),
        conv(256, (5, 5), name="layer_1", strides_down=2, use_bias=True,
             activation=tf.nn.relu),
        conv(HYPERPRIOR_DEPTH, (5, 5), name="layer_2", strides_down=2,
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
        kernel_parameterizer=None, activation=tf.nn.relu)

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

  def __init__(self):
    super().__init__()
    conv = functools.partial(
        tfc.SignalConv2D, corr=False, strides_up=1, padding="same_zeros",
        use_bias=True, kernel_parameterizer=None)

    # Note that the number of channels in the output tensor must match the
    # size of the corresponding slice. If we have 10 slices and a bottleneck
    # with 320 channels, the output is 320 / 10 = 32 channels.
    slice_depth = LATENT_DEPTH // NUM_SLICES
    if slice_depth * NUM_SLICES != LATENT_DEPTH:
      raise ValueError("Slices do not evenly divide latent depth (%d / %d)" % (
          LATENT_DEPTH, NUM_SLICES))

    self.transform = tf.keras.Sequential([
        conv(224, (5, 5), name="layer_0", activation=tf.nn.relu),
        conv(128, (5, 5), name="layer_1", activation=tf.nn.relu),
        conv(slice_depth, (3, 3), name="layer_2", activation=None),
    ])

  def call(self, tensor):
    return self.transform(tensor)


class CompressionModel(tf.Module):
  """Module that encapsulates the compression model."""

  def __init__(self, args):
    super().__init__()
    self.args = args

    self.analysis_transform = AnalysisTransform()
    self.synthesis_transform = SynthesisTransform()
    self.hyper_analysis_transform = HyperAnalysisTransform()
    self.hyper_synthesis_mean_transform = HyperSynthesisTransform()
    self.hyper_synthesis_scale_transform = HyperSynthesisTransform()
    self.cc_mean_transforms = [SliceTransform() for _ in range(NUM_SLICES)]
    self.cc_scale_transforms = [SliceTransform() for _ in range(NUM_SLICES)]
    self.lrp_transforms = [SliceTransform() for _ in range(NUM_SLICES)]
    self.entropy_bottleneck = tfc.NoisyDeepFactorized(
        batch_shape=[HYPERPRIOR_DEPTH])
    self.optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

    tf.summary.experimental.set_step(self.optimizer.iterations)
    self.writer = tf.summary.create_file_writer(args.checkpoint_dir)

  def train_one_step(self, x):
    """Build model and apply gradients."""
    with tf.GradientTape() as tape, self.writer.as_default():
      loss = self._run("train", x=x)

    grads = tape.gradient(loss, self.trainable_variables)
    self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
    self.writer.flush()
    return loss

  def compress(self, x):
    """Build model and compress latents."""
    mse, bpp, x_hat, pack = self._run("compress", x=x)

    # Write a binary file with the shape information and the compressed string.
    packed = tfc.PackedTensors()
    tensors, arrays = zip(*pack)
    packed.pack(tensors, arrays)
    with open(self.args.output_file, "wb") as f:
      f.write(packed.string)

    # If requested, transform the quantized image back and measure performance.
    if self.args.verbose:
      x *= 255  # x_hat is already in the [0..255] range
      psnr = tf.squeeze(tf.image.psnr(x_hat, x, 255))
      msssim = tf.squeeze(tf.image.ssim_multiscale(x_hat, x, 255))

      # The actual bits per pixel including overhead.
      x_shape = tf.shape(x)
      num_pixels = tf.cast(tf.reduce_prod(x_shape[:-1]), dtype=tf.float32)
      packed_bpp = len(packed.string) * 8 / num_pixels

      print("Mean squared error: {:0.4f}".format(mse))
      print("PSNR (dB): {:0.2f}".format(psnr))
      print("Multiscale SSIM: {:0.4f}".format(msssim))
      print("Multiscale SSIM (dB): {:0.2f}".format(-10 * np.log10(1 - msssim)))
      print("Information content in bpp: {:0.4f}".format(bpp))
      print("Actual bits per pixel: {:0.4f}".format(packed_bpp))

    return x_hat

  def decompress(self, bit_strings):
    """Build model and decompress bitstrings to generate a reconstruction."""
    return self._run("decompress", bit_strings=bit_strings)

  def _run(self, mode, x=None, bit_strings=None):
    """Run model according to `mode` (train, compress, or decompress)."""
    training = (mode == "train")

    if mode == "decompress":
      x_shape, y_shape, z_shape, z_string = bit_strings[:4]
      y_strings = bit_strings[4:]
      assert len(y_strings) == NUM_SLICES
    else:
      y_strings = []
      x_shape = tf.shape(x)[1:-1]

      # Build the encoder (analysis) half of the hierarchical autoencoder.
      y = self.analysis_transform(x)
      y_shape = tf.shape(y)[1:-1]

      z = self.hyper_analysis_transform(y)
      z_shape = tf.shape(z)[1:-1]

    if mode == "train":
      num_pixels = self.args.batchsize * self.args.patchsize ** 2
    else:
      num_pixels = tf.cast(tf.reduce_prod(x_shape), tf.float32)

    # Build the entropy model for the hyperprior (z).
    em_z = tfc.ContinuousBatchedEntropyModel(
        self.entropy_bottleneck, coding_rank=3,
        compression=not training, no_variables=True)

    if mode != "decompress":
      # When training, z_bpp is based on the noisy version of z (z_tilde).
      _, z_bits = em_z(z, training=training)
      z_bpp = tf.reduce_mean(z_bits) / num_pixels

    if training:
      # Use rounding (instead of uniform noise) to modify z before passing it
      # to the hyper-synthesis transforms. Note that quantize() overrides the
      # gradient to create a straight-through estimator.
      z_hat = em_z.quantize(z)
      z_string = None
    else:
      if mode == "compress":
        z_string = em_z.compress(z)
      z_hat = em_z.decompress(z_string, z_shape)

    # Build the decoder (synthesis) half of the hierarchical autoencoder.
    latent_scales = self.hyper_synthesis_scale_transform(z_hat)
    latent_means = self.hyper_synthesis_mean_transform(z_hat)

    # En/Decode each slice conditioned on hyperprior and previous slices.
    y_slices = (y_strings if mode == "decompress" else
                tf.split(y, NUM_SLICES, axis=-1))
    y_hat_slices = []
    y_bpps = []
    for slice_index, y_slice in enumerate(y_slices):
      # Model may condition on only a subset of previous slices.
      support_slices = (y_hat_slices if MAX_SUPPORT_SLICES < 0 else
                        y_hat_slices[:MAX_SUPPORT_SLICES])

      # Predict mu and sigma for the current slice.
      mean_support = tf.concat([latent_means] + support_slices, axis=-1)
      mu = self.cc_mean_transforms[slice_index](mean_support)
      mu = mu[:, :y_shape[0], :y_shape[1], :]

      # Note that in this implementation, `sigma` represents scale indices,
      # not actual scale values.
      scale_support = tf.concat([latent_scales] + support_slices, axis=-1)
      sigma = self.cc_scale_transforms[slice_index](scale_support)
      sigma = sigma[:, :y_shape[0], :y_shape[1], :]

      # Build the conditional entropy model for this slice.
      em_y = tfc.LocationScaleIndexedEntropyModel(
          tfc.NoisyNormal, num_scales=SCALES_LEVELS, scale_fn=scale_fn,
          coding_rank=3, compression=not training, no_variables=True)

      if mode == "decompress":
        y_hat_slice = em_y.decompress(y_slice, sigma, loc=mu)
      else:
        _, slice_bits = em_y(y_slice, sigma, loc=mu, training=training)
        slice_bpp = tf.reduce_mean(slice_bits) / num_pixels
        y_bpps.append(slice_bpp)

        if training:
          # For the synthesis transform, use rounding. Note that quantize()
          # overrides the gradient to create a straight-through estimator.
          y_hat_slice = em_y.quantize(y_slice, sigma, loc=mu)
        else:
          assert mode == "compress"
          slice_string = em_y.compress(y_slice, sigma, mu)
          y_strings.append(slice_string)
          y_hat_slice = em_y.decompress(slice_string, sigma, mu)

      # Add latent residual prediction (LRP).
      lrp_support = tf.concat([mean_support, y_hat_slice], axis=-1)
      lrp = self.lrp_transforms[slice_index](lrp_support)
      lrp = 0.5 * tf.math.tanh(lrp)
      y_hat_slice += lrp

      y_hat_slices.append(y_hat_slice)

    # Merge slices and generate the image reconstruction.
    y_hat = tf.concat(y_hat_slices, axis=-1)
    x_hat = self.synthesis_transform(y_hat)
    x_hat = x_hat[:, :x_shape[0], :x_shape[1], :]

    if mode != "decompress":
      # Total bpp is sum of bpp from hyperprior and all slices.
      total_bpp = tf.add_n(y_bpps + [z_bpp])

    # Mean squared error across pixels.
    if training:
      # Don't clip or round pixel values while training.
      mse = tf.reduce_mean(tf.math.squared_difference(x, x_hat))
      mse *= 255 ** 2  # multiply by 255^2 to correct for rescaling
    else:
      x_hat = tf.clip_by_value(x_hat, 0, 1)
      x_hat = tf.round(x_hat * 255)
      if mode == "compress":
        mse = tf.reduce_mean(tf.math.squared_difference(x * 255, x_hat))

    if mode == "train":
      # Calculate and return the rate-distortion loss: R + lambda * D.
      loss = total_bpp + self.args.lmbda * mse

      tf.summary.scalar("bpp", total_bpp)
      tf.summary.scalar("mse", mse)
      tf.summary.scalar("loss", loss)
      tf.summary.image("original", quantize_image(x))
      tf.summary.image("reconstruction", quantize_image(x_hat))

      return loss
    elif mode == "compress":
      # Create `pack` dict mapping tensors to values.
      tensors = [x_shape, y_shape, z_shape, z_string] + y_strings
      pack = [(v, v.numpy()) for v in tensors]
      return mse, total_bpp, x_hat, pack
    elif mode == "decompress":
      return x_hat


def train(args):
  """Trains the model."""
  if args.check_numerics:
    tf.debugging.enable_check_numerics()

  # Create input data pipeline.
  with tf.device("/cpu:0"):
    train_files = glob.glob(args.train_glob)
    if not train_files:
      raise RuntimeError(
          "No training images found with glob '{}'.".format(args.train_glob))
    train_dataset = tf.data.Dataset.from_tensor_slices(train_files)
    train_dataset = train_dataset.shuffle(buffer_size=len(train_files)).repeat()
    train_dataset = train_dataset.map(
        read_png, num_parallel_calls=args.preprocess_threads)
    train_dataset = train_dataset.map(
        lambda x: tf.image.random_crop(x, (args.patchsize, args.patchsize, 3)))
    train_dataset = train_dataset.batch(args.batchsize)
    train_dataset = train_dataset.prefetch(32)

  model = CompressionModel(args)
  step_counter = model.optimizer.iterations

  # Create checkpoint manager and restore the checkpoint if available.
  checkpoint = tf.train.Checkpoint(model=model)
  checkpoint_manager = tf.train.CheckpointManager(
      checkpoint, args.checkpoint_dir, max_to_keep=3,
      step_counter=step_counter, checkpoint_interval=args.checkpoint_interval)

  restore_path = tf.train.latest_checkpoint(args.checkpoint_dir)
  if restore_path:
    print("Last checkpoint:", restore_path)
    restore_status = checkpoint.restore(restore_path)
  else:
    restore_status = None

  # Run a simple training loop.
  for x in train_dataset:
    step = step_counter.numpy()
    loss = model.train_one_step(x)
    print("step %d:  train_loss=%f" % (step, loss))
    if restore_status is not None:
      restore_status.assert_consumed()
      restore_status = None

    finished = (step + 1 >= args.last_step)
    checkpoint_manager.save(check_interval=not finished)
    if finished: break


def compress(args):
  """Compresses an image."""
  # Load input image and add batch dimension.
  x = read_png(args.input_file)
  x = tf.expand_dims(x, 0)
  x.set_shape([1, None, None, 3])

  # Build model, restore optimized parameters, and compress the input image.
  model = CompressionModel(args)
  checkpoint = tf.train.Checkpoint(model=model)
  restore_path = tf.train.latest_checkpoint(args.checkpoint_dir)
  checkpoint.restore(restore_path)
  model.compress(x)


def decompress(args):
  """Decompresses an image."""
  # Three integers for tensor shapes + hyperprior and N slice strings.
  np_dtypes = [np.integer] * 3 + [np.bytes_] * (NUM_SLICES + 1)
  with open(args.input_file, "rb") as f:
    packed = tfc.PackedTensors(f.read())
  arrays = packed.unpack_from_np_dtypes(np_dtypes)

  # Build model, restore optimized parameters, and compress the input image.
  model = CompressionModel(args)

  checkpoint = tf.train.Checkpoint(model=model)
  restore_path = tf.train.latest_checkpoint(args.checkpoint_dir)
  print("Restore checkpoint:", restore_path)
  checkpoint.restore(restore_path)

  x_hat = model.decompress(arrays)

  # Write reconstructed image out as a PNG file.
  write_png(args.output_file, x_hat[0] / 255)


def parse_args(argv):
  """Parses command line arguments."""
  parser = argparse_flags.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)

  # High-level options.
  parser.add_argument(
      "--verbose", "-V", action="store_true",
      help="Report bitrate and distortion when training or compressing.")
  parser.add_argument(
      "--checkpoint_dir", default="train",
      help="Directory where to save/load model checkpoints.")

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
      description="Trains (or continues to train) a new model.")
  train_cmd.add_argument(
      "--train_glob", default="images/*.png",
      help="Glob pattern identifying training data. This pattern must expand "
           "to a list of RGB images in PNG format.")
  train_cmd.add_argument(
      "--batchsize", type=int, default=8,
      help="Batch size for training.")
  train_cmd.add_argument(
      "--patchsize", type=int, default=256,
      help="Size of image patches for training.")
  train_cmd.add_argument(
      "--lambda", type=float, default=0.01, dest="lmbda",
      help="Lambda for rate-distortion tradeoff.")
  train_cmd.add_argument(
      "--last_step", type=int, default=1000000,
      help="Train up to this number of steps.")
  train_cmd.add_argument(
      "--preprocess_threads", type=int, default=16,
      help="Number of CPU threads to use for parallel decoding of training "
           "images.")
  train_cmd.add_argument(
      "--checkpoint_interval", type=int, default=1000,
      help="Write a checkpoint every `checkpoint_interval` training steps.")
  train_cmd.add_argument(
      "--check_numerics", type=bool, default=True,
      help="Enable or disable TF support for catching NaN and inf in tensors.")

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
        help="Output filename (optional). If not provided, appends '{}' to "
             "the input filename.".format(ext))

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
