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
"""Some helper enums and classes, as well as LPIPS downloader."""


import collections
import enum
import os
import pprint
import urllib.request


_LPIPS_URL = "http://rail.eecs.berkeley.edu/models/lpips/net-lin_alex_v0.1.pb"

TFDSArguments = collections.namedtuple(
    "TFDSArguments", ["dataset_name", "features_key", "downloads_dir"])


class ModelType(enum.Enum):
  # Train hyperprior model: encoder/decoder/entropy model.
  COMPRESSION = "compression"
  # Additionally train a discriminator, and use GAN loss.
  COMPRESSION_GAN = "compression_gan"


class ModelMode(enum.Enum):
  TRAINING = "training"
  VALIDATION = "validation"
  EVALUATION = "evaluation"


class Config(object):
  """Lightweight class to enable dot-notation on dictionaries."""

  def __init__(self, **config):
    self._config = config

  def __getattr__(self, a):
    return self._config[a]

  def __getitem__(self, i):
    return self._config[i]

  def __repr__(self):
    return f"Config({self._config})"

  def __str__(self):
    return "Config(\n" + pprint.pformat(self._config) + ")"


def ensure_lpips_weights_exist(weight_path_out):
  """Downloads weights if needed."""
  if os.path.isfile(weight_path_out):
    return
  print("Downloading LPIPS weights:", _LPIPS_URL, "->", weight_path_out)
  urllib.request.urlretrieve(_LPIPS_URL, weight_path_out)
  if not os.path.isfile(weight_path_out):
    raise ValueError(f"Failed to download LPIPS weights from {_LPIPS_URL} "
                     f"to {weight_path_out}. Please manually download!")


def add_tfds_arguments(parser):
  parser.add_argument(
      "--tfds_dataset_name", default="coco2014", help="TFDS dataset name.")
  parser.add_argument(
      "--tfds_downloads_dir",
      default=None,
      help=("Where TFDS stores data."
            "Defaults to ~/tensorflow_datasets."))
  parser.add_argument(
      "--tfds_features_key",
      default="image",
      help="Name of the TFDS feature to use.")


def parse_tfds_arguments(args) -> TFDSArguments:
  return TFDSArguments(args.tfds_dataset_name, args.tfds_features_key,
                       args.tfds_downloads_dir)
