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

# This script must run at the workspace root directory.

set -ex  # Fail if any command fails, echo commands.

# Script configuration --------------------------------------------------------
OUTPUT_DIR="${1-/tmp/tensorflow_compression_ops}"
WHEEL_VERSION=${2-0.dev0}

# Optionally exported environment variables.
: ${BAZEL_OPT:=}
# -----------------------------------------------------------------------------

python -m pip install -U pip setuptools wheel
python -m pip install -r tensorflow_compression_ops/requirements.txt
bazel build ${BAZEL_OPT} -c opt --copt=-mavx tensorflow_compression/cc:libtensorflow_compression.so

SRCDIR="$(mktemp -d)"
trap 'rm -r -- "${SRCDIR}"' EXIT

cp LICENSE "${SRCDIR}"
cp tensorflow_compression_ops/README.md "${SRCDIR}"
cp tensorflow_compression_ops/MANIFEST.in "${SRCDIR}"
cp tensorflow_compression_ops/requirements.txt "${SRCDIR}"

mkdir -p "${SRCDIR}/tensorflow_compression_ops"
cp tensorflow_compression_ops/__init__.py "${SRCDIR}/tensorflow_compression_ops/__init__.py"

mkdir -p "${SRCDIR}/tensorflow_compression_ops/cc"
cp "$(bazel info -c opt bazel-genfiles)/tensorflow_compression/cc/libtensorflow_compression.so" \
  "${SRCDIR}/tensorflow_compression_ops/cc"

mkdir -p "${SRCDIR}/tensorflow_compression_ops/tests"
touch "${SRCDIR}/tensorflow_compression_ops/tests/__init__.py"
cp tensorflow_compression_ops/tests_all.py "${SRCDIR}/tensorflow_compression_ops/tests/all.py"
sed -e "s/from tensorflow_compression.python.ops import gen_ops/import tensorflow_compression_ops as gen_ops/" \
  tensorflow_compression/python/ops/quantization_ops_test.py \
  > "${SRCDIR}/tensorflow_compression_ops/tests/quantization_ops_test.py"
sed -e "s/from tensorflow_compression.python.ops import gen_ops/import tensorflow_compression_ops as gen_ops/" \
  tensorflow_compression/python/ops/range_coding_ops_test.py \
  > "${SRCDIR}/tensorflow_compression_ops/tests/range_coding_ops_test.py"

python tensorflow_compression_ops/build_pip_pkg.py "${SRCDIR}" "${OUTPUT_DIR}" "${WHEEL_VERSION}"
