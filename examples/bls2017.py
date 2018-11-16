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

Modified by Victor Xing (11/15/2018)
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

# Dependency imports

import numpy as np
import tensorflow as tf
import tensorflow_compression as tfc

import glob
import functools
import time
import os
import shutil
import sys
from tensorflow.python.client import timeline

# Parameters used in the training phase, change to your preference
EVAL_FREQUENCY = 500
MDATA_FREQUENCY = 500
SAVE_FREQUENCY = 1000
VISUALIZE_FREQUENCY = 10000

# Your number of processors, for parallel preprocessing
NUM_THREADS = 48


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


def train_preprocess(image):
  """Performs data augmentation on the training set."""

  image = tf.image.random_flip_left_right(image)
  image = tf.image.random_flip_up_down(image)
  image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
  image = tf.image.random_brightness(image, max_delta=32.0 / 255.0)
  image = tf.image.random_saturation(image, lower=0.8, upper=1.2)
  image = tf.image.random_hue(image, max_delta=0.2)

  # Random cropping
  crop_shape = (args.patchsize, args.patchsize, 3)
  image = tf.random_crop(image, crop_shape)

  # Make sure the image is still in [0, 1]
  image = tf.clip_by_value(image, 0.0, 1.0)
  return image


def train():
  """Trains the model."""

  if args.verbose:
    tf.logging.set_verbosity(tf.logging.INFO)

  # Create input data pipeline.
  with tf.device('/cpu:0'):
    train_files = glob.glob(args.train_glob)
    train_dataset = tf.data.Dataset.from_tensor_slices(train_files)
    train_dataset = train_dataset.apply(
                      tf.contrib.data.shuffle_and_repeat(len(train_files)))
    train_dataset = train_dataset.map(load_image,
                                      num_parallel_calls=NUM_THREADS)
    train_dataset = train_dataset.map(train_preprocess,
                                      num_parallel_calls=NUM_THREADS)
    train_dataset = train_dataset.batch(args.batchsize)
    train_dataset = train_dataset.prefetch(1)

    # Need to crop validation and test images too to have the same size
    # Can probably use deterministic instead of random cropping...
    mapcrop = functools.partial(tf.random_crop,
                                size=(args.patchsize, args.patchsize, 3))

    valid_files = glob.glob(args.valid_glob)
    valid_dataset = tf.data.Dataset.from_tensor_slices(valid_files)
    valid_dataset = valid_dataset.map(load_image,
                                      num_parallel_calls=NUM_THREADS)
    valid_dataset = valid_dataset.map(mapcrop,
                                      num_parallel_calls=NUM_THREADS)
    valid_dataset = valid_dataset.batch(args.batchsize)

    test_files = glob.glob(args.test_glob)
    test_dataset = tf.data.Dataset.from_tensor_slices(test_files)
    test_dataset = test_dataset.map(load_image,
                                    num_parallel_calls=NUM_THREADS)
    test_dataset = test_dataset.map(mapcrop,
                                    num_parallel_calls=NUM_THREADS)
    test_dataset = test_dataset.batch(args.batchsize)

    num_pixels = args.patchsize**2 * args.batchsize

  # Build reinitializable initializers.
  iter = tf.data.Iterator.from_structure(train_dataset.output_types,
                                         train_dataset.output_shapes)
  x = iter.get_next()
  train_init_op = iter.make_initializer(train_dataset)
  valid_init_op = iter.make_initializer(valid_dataset)
  test_init_op = iter.make_initializer(test_dataset)

  # Build autoencoder.
  y = analysis_transform(x, args.num_filters)
  entropy_bottleneck = tfc.EntropyBottleneck()
  y_tilde, likelihoods = entropy_bottleneck(y, training=True)
  x_tilde = synthesis_transform(y_tilde, args.num_filters)

  # Total number of bits divided by number of pixels.
  bpp = tf.reduce_sum(tf.log(likelihoods)) / (-np.log(2) * num_pixels)

  # Mean squared error across pixels.
  mse = tf.reduce_sum(tf.squared_difference(x, x_tilde))
  # Multiply by 255^2 to correct for rescaling.
  mse *= 255 ** 2 / num_pixels

  # The rate-distortion cost.
  loss = args.lmbda * mse + bpp

  # Minimize loss and auxiliary loss, and execute update op.
  step = tf.train.create_global_step()
  main_optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)
  main_step = main_optimizer.minimize(loss, global_step=step)

  aux_optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)
  aux_step = aux_optimizer.minimize(entropy_bottleneck.losses[0])

  train_op = tf.group(main_step, aux_step, entropy_bottleneck.updates[0])

  saver = tf.train.Saver()

  with tf.Session() as sess:
    # Initialize summaries for Tensorboard
    # Plots: training and validation loss, training and validation rate (bpp)
    loss_summ = tf.summary.scalar("loss", loss)
    bpp_summ = tf.summary.scalar("bpp", bpp)
    val_loss, val_loss_update = tf.metrics.mean(loss, name="val_loss")
    val_bpp, val_bpp_update = tf.metrics.mean(bpp, name="val_bpp")
    val_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES,
                                 scope="val_loss") \
      + tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES,
                          scope="val_bpp")
    val_loss_summ = tf.summary.scalar("loss", val_loss)
    val_bpp_summ = tf.summary.scalar("bpp", val_bpp)
    merged = tf.summary.merge([loss_summ, bpp_summ, val_loss_summ,
                               val_bpp_summ])

    # Initialize graph parameters
    sess.run(tf.variables_initializer(var_list=val_vars))
    sess.run(tf.global_variables_initializer())
    print('\nVariables initialized\n')
    print('Number of trainable parameters : {:d}\n'
          .format(np.sum([np.prod(v.get_shape().as_list())
                         for v in tf.trainable_variables()])))

    n_train = len(glob.glob(args.train_glob))
    n_valid = len(glob.glob(args.valid_glob))
    train_writer = tf.summary.FileWriter(train_log_dir, sess.graph)
    val_writer = tf.summary.FileWriter(val_log_dir)
    step = 0
    start_time = epoch_time = time.time()

    # Begin training loop
    for epoch in range(args.epochs):
      epoch_time = time.time()
      sess.run(train_init_op)
      for _ in range(n_train // args.batchsize):
        step += 1
        if step % MDATA_FREQUENCY == MDATA_FREQUENCY - 1:
            # Regularly log runtime statistics
            # Access it in Tensorboard in the Graph tab, "Session runs"
            # dropdown menu
            run_options = tf.RunOptions(
                            trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()
            summary, _ = sess.run([merged, train_op],
                                  options=run_options,
                                  run_metadata=run_metadata)
            train_writer.add_run_metadata(run_metadata,
                                          'step%03d' % step)
            train_writer.add_summary(summary, step)
        else:
            summary, _ = sess.run([merged, train_op])
            train_writer.add_summary(summary, step)
        if step % VISUALIZE_FREQUENCY == VISUALIZE_FREQUENCY - 1:
            # Visualize the current estimated pdf of \tilde{y} in Tensorboard
            # (Images tab)
            vis_img = sess.run(entropy_bottleneck.visualize())
            train_writer.add_summary(vis_img, step)
        if step % SAVE_FREQUENCY == 0:
            saver.save(sess, args.checkpoint_dir+'/model', global_step=step)

      # Compute validation loss and bpp
      sess.run(valid_init_op)
      sess.run(tf.variables_initializer(var_list=val_vars))
      for _ in range(n_valid // args.batchsize):
         sess.run([val_loss_update, val_bpp_update])
      summary = sess.run(val_loss_summ)
      val_writer.add_summary(summary, step)
      summary = sess.run(val_bpp_summ)
      val_writer.add_summary(summary, step)
      elapsed_time = time.time() - epoch_time

      # Print info on training progression
      print('Iteration %d (epoch %d)' % (step, epoch))
      print('Training loss: %.3f' % sess.run(loss))
      print('Validation loss: %.1f' % sess.run(val_loss))
      print('%.1f s\n' % elapsed_time)
      sys.stdout.flush()

    training_time = time.time() - start_time
    train_writer.close()
    val_writer.close()

    # Print the test loss and total training time
    sess.run(test_init_op)
    sess.run(tf.variables_initializer(var_list=val_vars))
    n_test = len(glob.glob(args.test_glob))
    for _ in range(n_test // args.batchsize):
        sess.run([val_loss_update, val_bpp_update])
    print('Test loss: %.1f' % sess.run(val_loss))
    print('Training time : %.1fs' % (training_time))


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

    # If requested, write a Timeline object that profiles runtime performance
    # The .json file can be read using chrome://tracing
    if args.profiling_comp:
      run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
      run_metadata = tf.RunMetadata()
      string, x_shape, y_shape = sess.run([string, tf.shape(x), tf.shape(y)],
                                          options=run_options,
                                          run_metadata=run_metadata)
      tl = timeline.Timeline(run_metadata.step_stats)
      ctf = tl.generate_chrome_trace_format()
      with open(os.path.join(args.log_dir, 'timeline.json'), 'w') as f:
        f.write(ctf)
    else:
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

      psnr = sess.run(tf.image.psnr(x_hat, x*255, 255))
      msssim = sess.run(tf.image.ssim_multiscale(x_hat, x*255, 255))

      print("Mean squared error: {:0.4}".format(mse))
      print("Information content of this image in bpp: {:0.4}".format(eval_bpp))
      print("Actual bits per pixel for this image: {:0.4}".format(bpp))
      print("PSNR (dB) : {:0.4}".format(psnr[0]))
      print("MS-SSIM (dB) : {:0.4}".format(-10*np.log10(1-msssim[0])))


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
      "--log_dir", default="log_dir",
      help="Directory where to save Tensorboard logs.")
  parser.add_argument(
      "--train_glob", default="images/*.png",
      help="Glob pattern identifying training data. This pattern must expand "
           "to a list of RGB images in PNG format which all have the same "
           "shape.")
  parser.add_argument(
      "--valid_glob", default="valid_imgs/*.png",
      help="Glob pattern identifying validation data. This pattern must expand "
           "to a list of RGB images in PNG format which all have the same "
           "shape.")
  parser.add_argument(
      "--test_glob", default="test_imgs/*.png",
      help="Glob pattern identifying test data. This pattern must expand "
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
      "--epochs", type=int, default=200,
      help="Number of epochs for training.")
  parser.add_argument(
      "--profiling_comp", type=bool, default=False,
      help="If True, create .json runtime profiling file of the compression.")

  args = parser.parse_args()

  # Create log directory for Tensorboard and overwrite existing run if it exists
  train_log_dir = os.path.join(args.log_dir, 'training/')
  val_log_dir = os.path.join(args.log_dir, 'validation/')
  if os.path.exists(os.path.dirname(args.log_dir)):
    for file in os.listdir(args.log_dir):
      file_path = os.path.join(args.log_dir, file)
      try:
        if os.path.isfile(file_path):
          os.unlink(file_path)
        elif os.path.isdir(file_path):
          shutil.rmtree(file_path)
      except Exception as e:
        print(e)
    else:
        os.makedirs(os.path.dirname(args.log_dir))
    os.makedirs(os.path.dirname(train_log_dir))
    os.makedirs(os.path.dirname(val_log_dir))

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
