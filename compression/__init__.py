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

# pylint: disable=wildcard-import
from compression.python.layers.entropybottleneck import *
from tensorflow.contrib.coder.python.ops.coder_ops import *
# pylint: enable=wildcard-import

from tensorflow.python.util.all_util import remove_undocumented
remove_undocumented(__name__, [
    "EntropyBottleneck",
    "pmf_to_quantized_cdf", "range_decode", "range_encode",
])
