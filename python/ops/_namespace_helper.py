# -*- coding: utf-8 -*-
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
"""Helps importing C ops with a clean namespace."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


def get_ops(module):
  """Returns a dict of ops defined in a module by blacklisting internals."""
  ops = dict()
  for name in dir(module):
    if name.startswith("_"):
      continue
    if name.endswith("_eager_fallback"):
      continue
    if name in ("LIB_HANDLE", "OP_LIST", "deprecated_endpoints", "tf_export"):
      continue
    ops[name] = getattr(module, name)
  return ops
