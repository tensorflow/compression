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
"""Generated operations from C++."""

from tensorflow.python.framework import load_library
from tensorflow.python.platform import resource_loader

ops = load_library.load_op_library(resource_loader.get_path_to_datafile(
    "../../cc/libtensorflow_compression.so"))
globals().update({n: getattr(ops, n) for n in dir(ops)})

# pylint:disable=undefined-all-variable
__all__ = [
    "create_range_encoder",
    "create_range_decoder",
    "entropy_decode_channel",
    "entropy_decode_finalize",
    "entropy_decode_index",
    "entropy_encode_channel",
    "entropy_encode_finalize",
    "entropy_encode_index",
    "pmf_to_quantized_cdf",
]
# pylint:enable=undefined-all-variable
