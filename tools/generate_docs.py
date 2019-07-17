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
"""Generates docs for the TensorFlow compression library."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

from absl import app
from absl import flags

from tensorflow_docs.api_generator import generate_lib

import tensorflow_compression as tfc

FLAGS = flags.FLAGS


def main(_):
  doc_generator = generate_lib.DocGenerator(
      root_title="TensorFlow/compression",
      py_modules=[("tfc", tfc)],
      base_dir=os.path.dirname(tfc.__file__),
      private_map={
          "tfc.python.ops": ["gen_range_coding_ops", "namespace_helper"],
      },
      code_url_prefix="https://github.com/tensorflow/compression/tree/master/"
                      "tensorflow_compression",
      api_cache=False,
  )
  sys.exit(doc_generator.build(FLAGS.output_dir))


if __name__ == "__main__":
  flags.DEFINE_string(
      "output_dir", "/tmp/tensorflow_compression/api_docs/python",
      "Output directory.")

  app.run(main)
