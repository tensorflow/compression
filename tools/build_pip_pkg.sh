#!/usr/bin/env bash
# Copyright 2023 Google LLC. All Rights Reserved.
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
OUTPUT_DIR="${1-/tmp/tensorflow_compression}"
WHEEL_VERSION=${2-0.dev0}

# Optionally exported environment variables.
: ${BAZEL_OPT:=}
# -----------------------------------------------------------------------------

python -m pip install -U pip setuptools wheel
python -m pip install -r requirements.txt
bazel build ${BAZEL_OPT} -c opt --copt=-mavx tensorflow_compression/cc:libtensorflow_compression.so

SRCDIR="$(mktemp -d)"
trap 'rm -r -- "${SRCDIR}"' EXIT

cp LICENSE README.md MANIFEST.in requirements.txt "${SRCDIR}"

mkdir -p "${SRCDIR}/tensorflow_compression/cc"
cp "$(bazel info -c opt bazel-genfiles)/tensorflow_compression/cc/libtensorflow_compression.so" \
  "${SRCDIR}/tensorflow_compression/cc"


copy_file() {
  local FILENAME="${1#./}"
  local DST="${SRCDIR%/}/$(dirname "${FILENAME}")"
  mkdir -p "${DST}"
  cp "${FILENAME}" "${DST}"
}

copy_file "tensorflow_compression/__init__.py"
copy_file "tensorflow_compression/all_tests.py"

# Assumes no irregular characters in the filenames.
find tensorflow_compression/python -name "*.py" \
  | while read filename; do copy_file "${filename}"; done

python tools/build_pip_pkg.py "${SRCDIR}" "${OUTPUT_DIR}" "${WHEEL_VERSION}"
