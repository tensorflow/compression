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
"""Data compression tools."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports

from tensorflow.python.util.all_util import remove_undocumented

# pylint: disable=wildcard-import
from tensorflow_compression.python.layers.entropy_models import *
from tensorflow_compression.python.layers.gdn import *
from tensorflow_compression.python.layers.initializers import *
from tensorflow_compression.python.layers.parameterizers import *
from tensorflow_compression.python.layers.signal_conv import *
from tensorflow_compression.python.ops.coder_ops import *
from tensorflow_compression.python.ops.math_ops import *
from tensorflow_compression.python.ops.padding_ops import *
from tensorflow_compression.python.ops.spectral_ops import *
# pylint: enable=wildcard-import

remove_undocumented(__name__, [
    "EntropyBottleneck", "GDN", "IdentityInitializer", "Parameterizer",
    "StaticParameterizer", "RDFTParameterizer", "NonnegativeParameterizer",
    "SignalConv1D", "SignalConv2D", "SignalConv3D",
    "upper_bound", "lower_bound", "same_padding_for_kernel", "irdft_matrix",
    "pmf_to_quantized_cdf", "range_decode", "range_encode",
])
