#!/usr/bin/env bash
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

set -ex  # Fail if any command fails, echo commands.

WHEEL="${1}"

# `import tensorflow_compression` in the bazel root directory produces cryptic
# error messages, because Python ends up looking for .so files under the
# subdirectories in the src repo instead of Python module libraries. Changing
# the current directory helps avoid running tests inside the bazel root
# direcotory by accident.
pushd /tmp

python -m pip install -U pip setuptools wheel
python -m pip install "${WHEEL}"
python -m pip list -v

# Logs elements of tfc namespace and runs unit tests.
python -c "import tensorflow_compression_ops as tfc; print('\n'.join(sorted(dir(tfc))))"
python -m tensorflow_compression_ops.tests.all

popd  # /tmp
