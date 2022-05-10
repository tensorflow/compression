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
"""All Python tests for tensorflow_compression.

This is a convenience file to be included in the pip package.
"""

import tensorflow as tf

# pylint: disable=wildcard-import
from tensorflow_compression.python.datasets.y4m_dataset_test import *

from tensorflow_compression.python.distributions.deep_factorized_test import *
from tensorflow_compression.python.distributions.helpers_test import *
from tensorflow_compression.python.distributions.round_adapters_test import *
from tensorflow_compression.python.distributions.uniform_noise_test import *

from tensorflow_compression.python.entropy_models.continuous_batched_test import *
from tensorflow_compression.python.entropy_models.continuous_indexed_test import *
from tensorflow_compression.python.entropy_models.power_law_test import *
from tensorflow_compression.python.entropy_models.universal_test import *

from tensorflow_compression.python.layers.gdn_test import *
from tensorflow_compression.python.layers.initializers_test import *
from tensorflow_compression.python.layers.parameters_test import *
from tensorflow_compression.python.layers.signal_conv_test import *
from tensorflow_compression.python.layers.soft_round_test import *

from tensorflow_compression.python.ops.math_ops_test import *
from tensorflow_compression.python.ops.padding_ops_test import *
from tensorflow_compression.python.ops.range_coding_ops_test import *
from tensorflow_compression.python.ops.round_ops_test import *

from tensorflow_compression.python.util.packed_tensors_test import *
# pylint: enable=wildcard-import


if __name__ == "__main__":
  tf.test.main()
