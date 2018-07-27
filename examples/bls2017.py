# -*- coding: utf-8 -*-
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
"""Basic nonlinear transform coder for RGB images.

This is a close approximation of the image compression model of
Ballé, Laparra, Simoncelli (2017):
End-to-end optimized image compression
https://arxiv.org/abs/1611.01704
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

# Dependency imports

import numpy as np
import tensorflow as tf
import tensorflow_compression as tfc


def load_image(filename):
  """Loads a PNG image file."""

  string = tf.read_file(filename)
  image = tf.image.decode_image(string, channels=3)
  image = tf.cast(image, tf.float32)
  image /= 255
  return image


def save_image(filename, image):
  """Saves an image to a PNG file."""

  image = tf.clip_by_value(image, 0, 1)
  image = tf.round(image * 255)
  image = tf.cast(image, tf.uint8)
  string = tf.image.encode_png(image)
  return tf.write_file(filename, string)


def analysis_transform(tensor, num_filters):
  """Builds the analysis transform."""

  with tf.variable_scope("analysis"):
    with tf.variable_scope("layer_0"):
      layer = tfc.SignalConv2D(
          num_filters, (9, 9), corr=True, strides_down=4, padding="same_zeros",
          use_bias=True, activation=tfc.GDN())
      tensor = layer(tensor)

    with tf.variable_scope("layer_1"):
      layer = tfc.SignalConv2D(
          num_filters, (5, 5), corr=True, strides_down=2, padding="same_zeros",
          use_bias=True, activation=tfc.GDN())
      tensor = layer(tensor)

    with tf.variable_scope("layer_2"):
      layer = tfc.SignalConv2D(
          num_filters, (5, 5), corr=True, strides_down=2, padding="same_zeros",
          use_bias=False, activation=None)
      tensor = layer(tensor)

    return tensor


def synthesis_transform(tensor, num_filters):
  """Builds the synthesis transform."""

  with tf.variable_scope("synthesis"):
    with tf.variable_scope("layer_0"):
      layer = tfc.SignalConv2D(
          num_filters, (5, 5), corr=False, strides_up=2, padding="same_zeros",
          use_bias=True, activation=tfc.GDN(inverse=True))
      tensor = layer(tensor)

    with tf.variable_scope("layer_1"):
      layer = tfc.SignalConv2D(
          num_filters, (5, 5), corr=False, strides_up=2, padding="same_zeros",
          use_bias=True, activation=tfc.GDN(inverse=True))
      tensor = layer(tensor)

    with tf.variable_scope("layer_2"):
      layer = tfc.SignalConv2D(
          3, (9, 9), corr=False, strides_up=4, padding="same_zeros",
          use_bias=True, activation=None)
      tensor = layer(tensor)

    return tensor


def train():
  """Trains the model."""

  if args.verbose:
    tf.logging.set_verbosity(tf.logging.INFO)

  # Load all training images into a constant.
  images = tf.map_fn(
      load_image, tf.matching_files(args.data_glob),
      dtype=tf.float32, back_prop=False)
  with tf.Session() as sess:
    images = tf.constant(sess.run(images), name="images")

  # Training inputs are random crops out of the images tensor.
  crop_shape = (args.batchsize, args.patchsize, args.patchsize, 3)
  x = tf.random_crop(images, crop_shape)
  num_pixels = np.prod(crop_shape[:-1])

  # Build autoencoder.
  y = analysis_transform(x, args.num_filters)
  entropy_bottleneck = tfc.EntropyBottleneck()
  y_tilde, likelihoods = entropy_bottleneck(y, training=True)
  x_tilde = synthesis_transform(y_tilde, args.num_filters)

  # Total number of bits divided by number of pixels.
  train_bpp = tf.reduce_sum(tf.log(likelihoods)) / (-np.log(2) * num_pixels)

  # Mean squared error across pixels.
  train_mse = tf.reduce_sum(tf.squared_difference(x, x_tilde))
  # Multiply by 255^2 to correct for rescaling.
  train_mse *= 255 ** 2 / num_pixels

  # The rate-distortion cost.
  train_loss = args.lmbda * train_mse + train_bpp

  # Minimize loss and auxiliary loss, and execute update op.
  step = tf.train.create_global_step()
  main_optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)
  main_step = main_optimizer.minimize(train_loss, global_step=step)

  aux_optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)
  aux_step = aux_optimizer.minimize(entropy_bottleneck.losses[0])

  train_op = tf.group(main_step, aux_step, entropy_bottleneck.updates[0])

  logged_tensors = [
      tf.identity(train_loss, name="train_loss"),
      tf.identity(train_bpp, name="train_bpp"),
      tf.identity(train_mse, name="train_mse"),
  ]
  hooks = [
      tf.train.StopAtStepHook(last_step=args.last_step),
      tf.train.NanTensorHook(train_loss),
      tf.train.LoggingTensorHook(logged_tensors, every_n_secs=60),
  ]
  with tf.train.MonitoredTrainingSession(
      hooks=hooks, checkpoint_dir=args.checkpoint_dir) as sess:
    while not sess.should_stop():
      sess.run(train_op)


def compress():
  """Compresses an image."""

  # Load input image and add batch dimension.
  x = load_image(args.input)
  x = tf.expand_dims(x, 0)
  x.set_shape([1, None, None, 3])

  # Transform and compress the image, then remove batch dimension.
  y = analysis_transform(x, args.num_filters)
  entropy_bottleneck = tfc.EntropyBottleneck()
  string = entropy_bottleneck.compress(y)
  string = tf.squeeze(string, axis=0)

  # Transform the quantized image back (if requested).
  y_hat, likelihoods = entropy_bottleneck(y, training=False)
  x_hat = synthesis_transform(y_hat, args.num_filters)

  num_pixels = tf.to_float(tf.reduce_prod(tf.shape(x)[:-1]))

  # Total number of bits divided by number of pixels.
  eval_bpp = tf.reduce_sum(tf.log(likelihoods)) / (-np.log(2) * num_pixels)

  # Mean squared error across pixels.
  x_hat = tf.clip_by_value(x_hat, 0, 1)
  x_hat = tf.round(x_hat * 255)
  mse = tf.reduce_sum(tf.squared_difference(x * 255, x_hat)) / num_pixels

  with tf.Session() as sess:
    # Load the latest model checkpoint, get the compressed string and the tensor
    # shapes.
    latest = tf.train.latest_checkpoint(checkpoint_dir=args.checkpoint_dir)
    tf.train.Saver().restore(sess, save_path=latest)
    string, x_shape, y_shape = sess.run([string, tf.shape(x), tf.shape(y)])

    # Write a binary file with the shape information and the compressed string.
    with open(args.output, "wb") as f:
      f.write(np.array(x_shape[1:-1], dtype=np.uint16).tobytes())
      f.write(np.array(y_shape[1:-1], dtype=np.uint16).tobytes())
      f.write(string)

    # If requested, transform the quantized image back and measure performance.
    if args.verbose:
      eval_bpp, mse, num_pixels = sess.run([eval_bpp, mse, num_pixels])

      # The actual bits per pixel including overhead.
      bpp = (8 + len(string)) * 8 / num_pixels

      print("Mean squared error: {:0.4}".format(mse))
      print("Information content of this image in bpp: {:0.4}".format(eval_bpp))
      print("Actual bits per pixel for this image: {:0.4}".format(bpp))


def decompress():
  """Decompresses an image."""

  # Read the shape information and compressed string from the binary file.
  with open(args.input, "rb") as f:
    x_shape = np.frombuffer(f.read(4), dtype=np.uint16)
    y_shape = np.frombuffer(f.read(4), dtype=np.uint16)
    string = f.read()

  y_shape = [int(s) for s in y_shape] + [args.num_filters]

  # Add a batch dimension, then decompress and transform the image back.
  strings = tf.expand_dims(string, 0)
  entropy_bottleneck = tfc.EntropyBottleneck(dtype=tf.float32)
  y_hat = entropy_bottleneck.decompress(
      strings, y_shape, channels=args.num_filters)
  x_hat = synthesis_transform(y_hat, args.num_filters)

  # Remove batch dimension, and crop away any extraneous padding on the bottom
  # or right boundaries.
  x_hat = x_hat[0, :x_shape[0], :x_shape[1], :]

  # Write reconstructed image out as a PNG file.
  op = save_image(args.output, x_hat)

  # Load the latest model checkpoint, and perform the above actions.
  with tf.Session() as sess:
    latest = tf.train.latest_checkpoint(checkpoint_dir=args.checkpoint_dir)
    tf.train.Saver().restore(sess, save_path=latest)
    sess.run(op)


if __name__ == "__main__":
  parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)

  parser.add_argument(
      "command", choices=["train", "compress", "decompress"],
      help="What to do: 'train' loads training data and trains (or continues "
           "to train) a new model. 'compress' reads an image file (lossless "
           "PNG format) and writes a compressed binary file. 'decompress' "
           "reads a binary file and reconstructs the image (in PNG format). "
           "input and output filenames need to be provided for the latter "
           "two options.")
  parser.add_argument(
      "input", nargs="?",
      help="Input filename.")
  parser.add_argument(
      "output", nargs="?",
      help="Output filename.")
  parser.add_argument(
      "--verbose", "-v", action="store_true",
      help="Report bitrate and distortion when training or compressing.")
  parser.add_argument(
      "--num_filters", type=int, default=128,
      help="Number of filters per layer.")
  parser.add_argument(
      "--checkpoint_dir", default="train",
      help="Directory where to save/load model checkpoints.")
  parser.add_argument(
      "--data_glob", default="images/*.png",
      help="Glob pattern identifying training data. This pattern must expand "
           "to a list of RGB images in PNG format which all have the same "
           "shape.")
  parser.add_argument(
      "--batchsize", type=int, default=8,
      help="Batch size for training.")
  parser.add_argument(
      "--patchsize", type=int, default=128,
      help="Size of image patches for training.")
  parser.add_argument(
      "--lambda", type=float, default=0.1, dest="lmbda",
      help="Lambda for rate-distortion tradeoff.")
  parser.add_argument(
      "--last_step", type=int, default=1000000,
      help="Train up to this number of steps.")

  args = parser.parse_args()

  if args.command == "train":
    train()
  elif args.command == "compress":
    if args.input is None or args.output is None:
      raise ValueError("Need input and output filename for compression.")
    compress()
  elif args.command == "decompress":
    if args.input is None or args.output is None:
      raise ValueError("Need input and output filename for decompression.")
    decompress()
