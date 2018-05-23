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
"""Range coder operations."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf


_coder_ops = tf.load_op_library(os.path.join(
    tf.resource_loader.get_data_files_path(), "../../_coder_ops.so"))

# TODO(nmjohn): Find a way to do the below mapping in this module automatically.
pmf_to_quantized_cdf = _coder_ops.pmf_to_quantized_cdf
range_decode = _coder_ops.range_decode
range_encode = _coder_ops.range_encode
