# Copyright 2022 Google LLC. All Rights Reserved.
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
"""Generates API docs for the TensorFlow Compression library."""

import os

from absl import app
from absl import flags

from tensorflow_docs.api_generator import generate_lib
from tensorflow_docs.api_generator import public_api

import tensorflow_compression as tfc


_OUTPUT_DIR = flags.DEFINE_string(
    "output_dir", "/tmp/generated_docs",
    "Where to write the resulting docs.")
_CODE_URL_PREFIX = flags.DEFINE_string(
    "code_url_prefix",
    "https://github.com/tensorflow/compression/tree/master/tensorflow_compression",
    "The URL prefix for links to code.")
_SEARCH_HINTS = flags.DEFINE_bool(
    "search_hints", True,
    "Whether to include metadata search hints in the generated files.")
_SITE_PATH = flags.DEFINE_string(
    "site_path", "/api_docs/python",
    "Path prefix in _toc.yaml.")


def gen_api_docs():
  """Generates API docs for the TensorFlow Compression library."""
  doc_generator = generate_lib.DocGenerator(
      root_title="TensorFlow Compression",
      py_modules=[("tfc", tfc)],
      base_dir=os.path.dirname(tfc.__file__),
      code_url_prefix=_CODE_URL_PREFIX.value,
      search_hints=_SEARCH_HINTS.value,
      site_path=_SITE_PATH.value,
      callbacks=[public_api.explicit_package_contents_filter])
  doc_generator.build(_OUTPUT_DIR.value)
  print("Output docs to: ", _OUTPUT_DIR.value)


def main(_):
  gen_api_docs()


if __name__ == "__main__":
  app.run(main)
