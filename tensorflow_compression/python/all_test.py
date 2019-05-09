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
"""Tests for tensorflow_compression/python.

This is a convenience file to be included in PIP package.
No BUILD entry exists for this file on purpose.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

# pylint: disable=wildcard-import
from tensorflow_compression.python.layers.entropy_models_test import *
from tensorflow_compression.python.layers.gdn_test import *
from tensorflow_compression.python.layers.parameterizers_test import *
from tensorflow_compression.python.layers.signal_conv_test import *

from tensorflow_compression.python.ops.math_ops_test import *
from tensorflow_compression.python.ops.padding_ops_test import *
from tensorflow_compression.python.ops.range_coding_ops_test import *
from tensorflow_compression.python.ops.spectral_ops_test import *
# pylint: enable=wildcard-import


if __name__ == "__main__":
  tf.test.main()
