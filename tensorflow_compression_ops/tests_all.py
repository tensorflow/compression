# Copyright 2024 Google LLC. All Rights Reserved.
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
"""All Python tests for tensorflow_compression_ops.

This is a convenience file to be included in the pip package.
"""

import tensorflow as tf

# pylint: disable=wildcard-import
from tensorflow_compression_ops.tests.quantization_ops_test import *
from tensorflow_compression_ops.tests.range_coding_ops_test import *
# pylint: enable=wildcard-import


if __name__ == "__main__":
  tf.test.main()
