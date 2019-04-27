# -*- coding: utf-8 -*-
# Copyright 2019 Google LLC. All Rights Reserved.
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
"""Converts an image between PNG and TFCI formats."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os

from absl import app
from absl.flags import argparse_flags
import numpy as np
from six.moves import urllib
import tensorflow as tf

import tensorflow_compression as tfc  # pylint:disable=unused-import


def read_png(filename):
  """Creates graph to load a PNG image file."""
  string = tf.io.read_file(filename)
  image = tf.image.decode_image(string)
  image = tf.expand_dims(image, 0)
  return image


def write_png(filename, image):
  """Creates graph to write a PNG image file."""
  image = tf.squeeze(image, 0)
  if image.dtype.is_floating:
    image = tf.round(image)
  if image.dtype != tf.uint8:
    image = tf.saturate_cast(image, tf.uint8)
  string = tf.image.encode_png(image)
  return tf.io.write_file(filename, string)


def load_metagraph(model, url_prefix, metagraph_cache):
  """Loads and caches a trained model metagraph."""
  filename = os.path.join(metagraph_cache, model + ".metagraph")
  try:
    with tf.io.gfile.GFile(filename, "rb") as f:
      string = f.read()
  except tf.errors.NotFoundError:
    url = url_prefix + "/" + model + ".metagraph"
    try:
      request = urllib.request.urlopen(url)
      string = request.read()
    finally:
      request.close()
    tf.io.gfile.makedirs(os.path.dirname(filename))
    with tf.io.gfile.GFile(filename, "wb") as f:
      f.write(string)
  metagraph = tf.MetaGraphDef()
  metagraph.ParseFromString(string)
  tf.train.import_meta_graph(metagraph)
  return metagraph.signature_def


def instantiate_signature(signature_def):
  """Fetches tensors defined in a signature from the graph."""
  graph = tf.get_default_graph()
  inputs = {
      k: graph.get_tensor_by_name(v.name)
      for k, v in signature_def.inputs.items()
  }
  outputs = {
      k: graph.get_tensor_by_name(v.name)
      for k, v in signature_def.outputs.items()
  }
  return inputs, outputs


def compress(model, input_file, output_file, url_prefix, metagraph_cache):
  """Compresses a PNG file to a TFCI file."""
  if not output_file:
    output_file = input_file + ".tfci"

  with tf.Graph().as_default():
    # Load model metagraph.
    signature_defs = load_metagraph(model, url_prefix, metagraph_cache)
    inputs, outputs = instantiate_signature(signature_defs["sender"])

    # Just one input tensor.
    inputs = inputs["input_image"]
    # Multiple output tensors, ordered alphabetically, without names.
    outputs = [outputs[k] for k in sorted(outputs) if k.startswith("channel:")]

    # Run encoder.
    with tf.Session() as sess:
      feed_dict = {inputs: sess.run(read_png(input_file))}
      arrays = sess.run(outputs, feed_dict=feed_dict)

    # Pack data into tf.Example.
    example = tf.train.Example()
    example.features.feature["MD"].bytes_list.value[:] = [model]
    for i, (array, tensor) in enumerate(zip(arrays, outputs)):
      feature = example.features.feature[chr(i + 1)]
      if array.ndim != 1:
        raise RuntimeError("Unexpected tensor rank: {}.".format(array.ndim))
      if tensor.dtype.is_integer:
        feature.int64_list.value[:] = array
      elif tensor.dtype == tf.string:
        feature.bytes_list.value[:] = array
      else:
        raise RuntimeError(
            "Unexpected tensor dtype: '{}'.".format(tensor.dtype))

    # Write serialized tf.Example to disk.
    with tf.io.gfile.GFile(output_file, "wb") as f:
      f.write(example.SerializeToString())


def decompress(input_file, output_file, url_prefix, metagraph_cache):
  """Decompresses a TFCI file and writes a PNG file."""
  if not output_file:
    output_file = input_file + ".png"

  with tf.Graph().as_default():
    # Deserialize tf.Example from disk and determine model.
    with tf.io.gfile.GFile(input_file, "rb") as f:
      example = tf.train.Example()
      example.ParseFromString(f.read())
    model = example.features.feature["MD"].bytes_list.value[0]

    # Load model metagraph.
    signature_defs = load_metagraph(model, url_prefix, metagraph_cache)
    inputs, outputs = instantiate_signature(signature_defs["receiver"])

    # Multiple input tensors, ordered alphabetically, without names.
    inputs = [inputs[k] for k in sorted(inputs) if k.startswith("channel:")]
    # Just one output operation.
    outputs = write_png(output_file, outputs["output_image"])

    # Unpack data from tf.Example.
    arrays = []
    for i, tensor in enumerate(inputs):
      feature = example.features.feature[chr(i + 1)]
      np_dtype = tensor.dtype.as_numpy_dtype
      if tensor.dtype.is_integer:
        arrays.append(np.array(feature.int64_list.value, dtype=np_dtype))
      elif tensor.dtype == tf.string:
        arrays.append(np.array(feature.bytes_list.value, dtype=np_dtype))
      else:
        raise RuntimeError(
            "Unexpected tensor dtype: '{}'.".format(tensor.dtype))

    # Run decoder.
    with tf.Session() as sess:
      feed_dict = dict(zip(inputs, arrays))
      sess.run(outputs, feed_dict=feed_dict)


def list_models(url_prefix):
  url = url_prefix + "/models.txt"
  try:
    request = urllib.request.urlopen(url)
    print(request.read())
  finally:
    request.close()


def parse_args(argv):
  parser = argparse_flags.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)

  # High-level options.
  parser.add_argument(
      "--url_prefix",
      default="https://storage.googleapis.com/tensorflow_compression/"
              "metagraphs",
      help="URL prefix for downloading model metagraphs.")
  parser.add_argument(
      "--metagraph_cache",
      default="/tmp/tfc_metagraphs",
      help="Directory where to cache model metagraphs.")
  subparsers = parser.add_subparsers(
      title="commands", help="Invoke '<command> -h' for more information.")

  # 'compress' subcommand.
  compress_cmd = subparsers.add_parser(
      "compress",
      description="Reads a PNG file, compresses it using the given model, and "
                  "writes a TFCI file.")
  compress_cmd.set_defaults(
      f=compress,
      a=["model", "input_file", "output_file", "url_prefix", "metagraph_cache"])
  compress_cmd.add_argument(
      "model",
      help="Unique model identifier. See 'models' command for options.")

  # 'decompress' subcommand.
  decompress_cmd = subparsers.add_parser(
      "decompress",
      description="Reads a TFCI file, reconstructs the image using the model "
                  "it was compressed with, and writes back a PNG file.")
  decompress_cmd.set_defaults(
      f=decompress,
      a=["input_file", "output_file", "url_prefix", "metagraph_cache"])

  # Arguments for both 'compress' and 'decompress'.
  for cmd, ext in ((compress_cmd, ".tfci"), (decompress_cmd, ".png")):
    cmd.add_argument(
        "input_file",
        help="Input filename.")
    cmd.add_argument(
        "output_file", nargs="?",
        help="Output filename (optional). If not provided, appends '{}' to "
             "the input filename.".format(ext))

  # 'models' subcommand.
  models_cmd = subparsers.add_parser(
      "models",
      description="Lists available trained models. Requires an internet "
                  "connection.")
  models_cmd.set_defaults(f=list_models, a=["url_prefix"])

  # Parse arguments.
  return parser.parse_args(argv[1:])


if __name__ == "__main__":
  # Parse arguments and run function determined by subcommand.
  app.run(
      lambda args: args.f(**{k: getattr(args, k) for k in args.a}),
      flags_parser=parse_args)
