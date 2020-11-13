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
import collections
import glob
import itertools
import os
import sys

import numpy as np
from PIL import Image
import tensorflow.compat.v1 as tf
import tensorflow_compression as tfc

from . import configs
from . import helpers
from . import model


# Show custom tf.logging calls.
tf.logging.set_verbosity(tf.logging.INFO)


def eval_trained_model(config_name,
                       ckpt_dir,
                       out_dir,
                       images_glob,
                       tfds_arguments: helpers.TFDSArguments,
                       max_images=None):
  """Evaluate a trained model."""
  config = configs.get_config(config_name)
  hific = model.HiFiC(config, helpers.ModelMode.EVALUATION)

  # Note: Automatically uses the validation split for TFDS.
  dataset = hific.build_input(
      batch_size=1,
      crop_size=None,
      images_glob=images_glob,
      tfds_arguments=tfds_arguments)
  image_names = get_image_names(images_glob)
  iterator = tf.data.make_one_shot_iterator(dataset)
  get_next_image = iterator.get_next()
  input_image = get_next_image['input_image']
  output_image, bitstring = hific.build_model(**get_next_image)

  input_image = tf.cast(tf.round(input_image[0, ...]), tf.uint8)
  output_image = tf.cast(tf.round(output_image[0, ...]), tf.uint8)

  os.makedirs(out_dir, exist_ok=True)

  accumulated_metrics = collections.defaultdict(list)

  with tf.Session() as sess:
    hific.restore_trained_model(sess, ckpt_dir)
    hific.prepare_for_arithmetic_coding(sess)

    for i in itertools.count(0):
      if max_images and i == max_images:
        break
      try:
        inp_np, otp_np, bitstring_np = \
          sess.run([input_image, output_image, bitstring])

        h, w, c = inp_np.shape
        assert c == 3
        bpp = get_arithmetic_coding_bpp(
            bitstring, bitstring_np, num_pixels=h * w)

        metrics = {'psnr': get_psnr(inp_np, otp_np),
                   'bpp_real': bpp}

        metrics_str = ' / '.join(f'{metric}: {value:.5f}'
                                 for metric, value in metrics.items())
        print(f'Image {i: 4d}: {metrics_str}, saving in {out_dir}...')

        for metric, value in metrics.items():
          accumulated_metrics[metric].append(value)

        # Save images.
        name = image_names.get(i, f'img_{i:010d}')
        Image.fromarray(inp_np).save(
            os.path.join(out_dir, f'{name}_inp.png'))
        Image.fromarray(otp_np).save(
            os.path.join(out_dir, f'{name}_otp_{bpp:.3f}.png'))

      except tf.errors.OutOfRangeError:
        print('No more inputs.')
        break

  print('\n'.join(f'{metric}: {np.mean(values)}'
                  for metric, values in accumulated_metrics.items()))
  print('Done!')


def get_arithmetic_coding_bpp(bitstring, bitstring_np, num_pixels):
  """Calculate bitrate we obtain with arithmetic coding."""
  # TODO(fab-jul): Add `compress` and `decompress` methods.
  packed = tfc.PackedTensors()
  packed.pack(tensors=bitstring, arrays=bitstring_np)
  return len(packed.string) * 8 / num_pixels


def get_psnr(inp, otp):
  mse = np.mean(np.square(inp.astype(np.float32) - otp.astype(np.float32)))
  psnr = 20. * np.log10(255.) - 10. * np.log10(mse)
  return psnr


def get_image_names(images_glob):
  if not images_glob:
    return {}
  return {i: os.path.splitext(os.path.basename(p))[0]
          for i, p in enumerate(sorted(glob.glob(images_glob)))}


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

  parser.add_argument('--images_glob', help='If given, use TODO')

  helpers.add_tfds_arguments(parser)

  args = parser.parse_args(argv[1:])
  return args


def main(args):
  eval_trained_model(args.config, args.ckpt_dir, args.out_dir,
                     args.images_glob,
                     helpers.parse_tfds_arguments(args))


if __name__ == '__main__':
  main(parse_args(sys.argv))
