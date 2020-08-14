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
"""Converts an image between PNG and TFCI formats.

Use this script to compress images with pre-trained models as published. See the
'models' subcommand for a list of available models.
"""

import argparse
import os
import sys
import urllib

from absl import app
from absl.flags import argparse_flags
import tensorflow.compat.v1 as tf

import tensorflow_compression as tfc  # pylint:disable=unused-import

# Default URL to fetch metagraphs from.
URL_PREFIX = "https://storage.googleapis.com/tensorflow_compression/metagraphs"
# Default location to store cached metagraphs.
METAGRAPH_CACHE = "/tmp/tfc_metagraphs"


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


def load_cached(filename):
  """Downloads and caches files from web storage."""
  pathname = os.path.join(METAGRAPH_CACHE, filename)
  try:
    with tf.io.gfile.GFile(pathname, "rb") as f:
      string = f.read()
  except tf.errors.NotFoundError:
    url = URL_PREFIX + "/" + filename
    try:
      request = urllib.request.urlopen(url)
      string = request.read()
    finally:
      request.close()
    tf.io.gfile.makedirs(os.path.dirname(pathname))
    with tf.io.gfile.GFile(pathname, "wb") as f:
      f.write(string)
  return string


def import_metagraph(model):
  """Imports a trained model metagraph into the current graph."""
  string = load_cached(model + ".metagraph")
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


def compress_image(model, input_image):
  """Compresses an image array into a bitstring."""
  with tf.Graph().as_default():
    # Load model metagraph.
    signature_defs = import_metagraph(model)
    inputs, outputs = instantiate_signature(signature_defs["sender"])

    # Just one input tensor.
    inputs = inputs["input_image"]
    # Multiple output tensors, ordered alphabetically, without names.
    outputs = [outputs[k] for k in sorted(outputs) if k.startswith("channel:")]

    # Run encoder.
    with tf.Session() as sess:
      arrays = sess.run(outputs, feed_dict={inputs: input_image})

    # Pack data into bitstring.
    packed = tfc.PackedTensors()
    packed.model = model
    packed.pack(outputs, arrays)
    return packed.string


def compress(model, input_file, output_file, target_bpp=None, bpp_strict=False):
  """Compresses a PNG file to a TFCI file."""
  if not output_file:
    output_file = input_file + ".tfci"

  # Load image.
  with tf.Graph().as_default():
    with tf.Session() as sess:
      input_image = sess.run(read_png(input_file))
      num_pixels = input_image.shape[-2] * input_image.shape[-3]

  if not target_bpp:
    # Just compress with a specific model.
    bitstring = compress_image(model, input_image)
  else:
    # Get model list.
    models = load_cached(model + ".models")
    models = models.decode("ascii").split()

    # Do a binary search over all RD points.
    lower = -1
    upper = len(models)
    bpp = None
    best_bitstring = None
    best_bpp = None
    while bpp != target_bpp and upper - lower > 1:
      i = (upper + lower) // 2
      bitstring = compress_image(models[i], input_image)
      bpp = 8 * len(bitstring) / num_pixels
      is_admissible = bpp <= target_bpp or not bpp_strict
      is_better = (best_bpp is None or
                   abs(bpp - target_bpp) < abs(best_bpp - target_bpp))
      if is_admissible and is_better:
        best_bitstring = bitstring
        best_bpp = bpp
      if bpp < target_bpp:
        lower = i
      if bpp > target_bpp:
        upper = i
    if best_bpp is None:
      assert bpp_strict
      raise RuntimeError(
          "Could not compress image to less than {} bpp.".format(target_bpp))
    bitstring = best_bitstring

  # Write bitstring to disk.
  with tf.io.gfile.GFile(output_file, "wb") as f:
    f.write(bitstring)


def decompress(input_file, output_file):
  """Decompresses a TFCI file and writes a PNG file."""
  if not output_file:
    output_file = input_file + ".png"

  with tf.Graph().as_default():
    # Unserialize packed data from disk.
    with tf.io.gfile.GFile(input_file, "rb") as f:
      packed = tfc.PackedTensors(f.read())

    # Load model metagraph.
    signature_defs = import_metagraph(packed.model)
    inputs, outputs = instantiate_signature(signature_defs["receiver"])

    # Multiple input tensors, ordered alphabetically, without names.
    inputs = [inputs[k] for k in sorted(inputs) if k.startswith("channel:")]
    # Just one output operation.
    outputs = write_png(output_file, outputs["output_image"])

    # Unpack data.
    arrays = packed.unpack(inputs)

    # Run decoder.
    with tf.Session() as sess:
      sess.run(outputs, feed_dict=dict(zip(inputs, arrays)))


def list_models():
  url = URL_PREFIX + "/models.txt"
  try:
    request = urllib.request.urlopen(url)
    print(request.read().decode("utf-8"))
  finally:
    request.close()


def parse_args(argv):
  """Parses command line arguments."""
  parser = argparse_flags.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)

  # High-level options.
  parser.add_argument(
      "--url_prefix",
      default=URL_PREFIX,
      help="URL prefix for downloading model metagraphs.")
  parser.add_argument(
      "--metagraph_cache",
      default=METAGRAPH_CACHE,
      help="Directory where to cache model metagraphs.")
  subparsers = parser.add_subparsers(
      title="commands", dest="command",
      help="Invoke '<command> -h' for more information.")

  # 'compress' subcommand.
  compress_cmd = subparsers.add_parser(
      "compress",
      formatter_class=argparse.ArgumentDefaultsHelpFormatter,
      description="Reads a PNG file, compresses it using the given model, and "
                  "writes a TFCI file.")
  compress_cmd.add_argument(
      "model",
      help="Unique model identifier. See 'models' command for options. If "
           "'target_bpp' is provided, don't specify the index at the end of "
           "the model identifier.")
  compress_cmd.add_argument(
      "--target_bpp", type=float,
      help="Target bits per pixel. If provided, a binary search is used to try "
           "to match the given bpp as close as possible. In this case, don't "
           "specify the index at the end of the model identifier. It will be "
           "automatically determined.")
  compress_cmd.add_argument(
      "--bpp_strict", action="store_true",
      help="Try never to exceed 'target_bpp'. Ignored if 'target_bpp' is not "
           "set.")

  # 'decompress' subcommand.
  decompress_cmd = subparsers.add_parser(
      "decompress",
      formatter_class=argparse.ArgumentDefaultsHelpFormatter,
      description="Reads a TFCI file, reconstructs the image using the model "
                  "it was compressed with, and writes back a PNG file.")

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
  subparsers.add_parser(
      "models",
      formatter_class=argparse.ArgumentDefaultsHelpFormatter,
      description="Lists available trained models. Requires an internet "
                  "connection.")

  # Parse arguments.
  args = parser.parse_args(argv[1:])
  if args.command is None:
    parser.print_usage()
    sys.exit(2)
  return args


def main(args):
  # Command line can override these defaults.
  global URL_PREFIX, METAGRAPH_CACHE
  URL_PREFIX = args.url_prefix
  METAGRAPH_CACHE = args.metagraph_cache

  # Invoke subcommand.
  if args.command == "compress":
    compress(args.model, args.input_file, args.output_file,
             args.target_bpp, args.bpp_strict)
  if args.command == "decompress":
    decompress(args.input_file, args.output_file)
  if args.command == "models":
    list_models()


if __name__ == "__main__":
  app.run(main, flags_parser=parse_args)
