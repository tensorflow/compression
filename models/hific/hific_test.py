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
"""Test HiFiC."""

import contextlib
import tempfile

import numpy as np
import tensorflow.compat.v1 as tf

from . import configs
from . import helpers
from . import model
from . import train


class FakeHiFiC(model.HiFiC):
  """Fake class for testing."""

  def _get_dataset(self, batch_size, crop_size, images_glob,
                   tfds_arguments: helpers.TFDSArguments):
    zeros = np.zeros((batch_size, crop_size, crop_size, 3))
    return (tf.data.Dataset.from_tensor_slices(
        (zeros,)).repeat(128).batch(batch_size))


class HiFiCTest(tf.test.TestCase):
  """Test public repo."""

  def setUp(self):
    super(HiFiCTest, self).setUp()

    self._lpips_weight_path = 'test.weights'

  def test_input_pipeline(self):
    crop_size = 128
    hific = FakeHiFiC(
        configs.get_config('hific'), mode=helpers.ModelMode.TRAINING)
    ds = hific.build_input(
        batch_size=2,
        crop_size=crop_size,
        tfds_arguments=helpers.TFDSArguments(
            dataset_name='', features_key='', downloads_dir=''))
    iterator = tf.data.make_initializable_iterator(ds)
    ds_next = iterator.get_next()
    with tf.Session() as sess:
      sess.run(iterator.initializer)
      image = sess.run(ds_next)
    self.assertEqual(image['input_image'].shape[1:3], (crop_size, crop_size))

  def test_config(self):
    config = helpers.Config(foo=1, bar=helpers.Config(baz=2))
    self.assertEqual(config.foo, 1)
    self.assertEqual(config['foo'], 1)
    self.assertEqual(config.bar.baz, 2)


@contextlib.contextmanager
def _update_constants_for_testing(steps):
  steps_default_value = train.SAVE_CHECKPOINT_STEPS
  train.SAVE_CHECKPOINT_STEPS = steps
  yield
  train.SAVE_CHECKPOINT_STEPS = steps_default_value


if __name__ == '__main__':
  tf.test.main()
