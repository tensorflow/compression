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
"""Data compression tools."""

from __future__ import absolute_import as _absolute_import
from __future__ import division as _division
from __future__ import print_function as _print_function

try:
  import tensorflow as _tensorflow
  _tf_version = [int(v) for v in _tensorflow.version.VERSION.split(".")]
  assert _tf_version[0] == 1 and _tf_version[1] >= 14
except (ImportError, AssertionError):
  raise RuntimeError("For tensorflow_compression, please install TensorFlow "
                     "1.14 or above. TensorFlow 2 is not yet supported.")


# pylint: disable=wildcard-import
from tensorflow_compression.python.layers.entropy_models import *
from tensorflow_compression.python.layers.gdn import *
from tensorflow_compression.python.layers.initializers import *
from tensorflow_compression.python.layers.parameterizers import *
from tensorflow_compression.python.layers.signal_conv import *
from tensorflow_compression.python.ops.math_ops import *
from tensorflow_compression.python.ops.padding_ops import *
from tensorflow_compression.python.ops.range_coding_ops import *
from tensorflow_compression.python.ops.spectral_ops import *
# pylint: enable=wildcard-import
