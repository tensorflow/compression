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
"""Eval models trained with train.py.

NOTE: To evaluate models used in the paper, use tfci.py! See README.md.
"""

import argparse
import itertools
import os
import sys
from PIL import Image

import tensorflow.compat.v1 as tf

from . import configs
from . import helpers
from . import model


def eval_trained_model(config_name,
                       ckpt_dir,
                       out_dir,
                       tfds_arguments: helpers.TFDSArguments,
                       max_images=None):
  """Evaluate a trained model."""
  config = configs.get_config(config_name)
  hific = model.HiFiC(config, helpers.ModelMode.EVALUATION)

  # Automatically uses the validation split.
  dataset = hific.build_input(
      batch_size=1, crop_size=None, tfds_arguments=tfds_arguments)
  iterator = tf.data.make_one_shot_iterator(dataset)
  get_next_image = iterator.get_next()

  output_image, bpp = hific.build_model(**get_next_image)
  input_image = get_next_image['input_image']

  input_image = tf.cast(tf.round(input_image[0, ...]), tf.uint8)
  output_image = tf.cast(tf.round(output_image[0, ...]), tf.uint8)

  os.makedirs(out_dir, exist_ok=True)

  with tf.Session() as sess:
    hific.restore_trained_model(sess, ckpt_dir)
    for i in itertools.count(0):
      if max_images and i == max_images:
        break
      try:
        inp_np, otp_np, bpp_np = sess.run([input_image, output_image, bpp])
        print(f'Image {i}: {bpp_np:.3} bpp, saving in {out_dir}...')
        Image.fromarray(inp_np).save(
            os.path.join(out_dir, f'img_{i:010d}inp.png'))
        Image.fromarray(otp_np).save(
            os.path.join(out_dir, f'img_{i:010d}otp_{bpp_np:.3f}.png'))
      except tf.errors.OutOfRangeError:
        print('No more inputs')
        break
  print('Done!')


def parse_args(argv):
  """Parses command line arguments."""
  parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('--config', required=True,
                      choices=configs.valid_configs(),
                      help='The config to use.')
  parser.add_argument('--ckpt_dir', required=True,
                      help=('Path to the folder where checkpoints of the '
                            'trained model are.'))
  parser.add_argument('--out_dir', required=True, help='Where to save outputs.')

  helpers.add_tfds_arguments(parser)

  args = parser.parse_args(argv[1:])
  return args


def main(args):
  eval_trained_model(args.config, args.ckpt_dir, args.out_dir,
                     helpers.parse_tfds_arguments(args))


if __name__ == '__main__':
  main(parse_args(sys.argv))
