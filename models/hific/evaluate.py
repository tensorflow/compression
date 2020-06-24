# Lint as: python3
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

from absl import app

import tensorflow.compat.v1 as tf

from . import configs
from . import helpers
from . import model


def eval_trained_model(config_name, ckpt_dir, max_images=None):
  """Evaluate a trained model."""
  config = configs.get_config(config_name)
  hific = model.HiFiC(config, helpers.ModelMode.EVALUATION)

  # Automatically uses the validation split of LSUN.
  dataset = hific.build_input(batch_size=1, crop_size=None, tfds_name='lsun')
  iterator = tf.data.make_one_shot_iterator(dataset)
  get_next_image = iterator.get_next()

  output_image, bpp = hific.build_model(**get_next_image)
  input_image = get_next_image['input_image']
  with tf.Session() as sess:
    hific.restore_trained_model(sess, ckpt_dir)
    for i in itertools.count(0):
      if max_images and i == max_images:
        break
      try:
        input_, output_, bpp_ = sess.run([input_image, output_image, bpp])
        # TODO(fab-jul): Save image, report bpp, etc.
        print(input_.shape, output_.shape, bpp_)
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
  args = parser.parse_args(argv[1:])
  return args


def main(args):
  eval_trained_model(args.config, args.ckpt_dir)


if __name__ == '__main__':
  app.run(main, flags_parser=parse_args)
