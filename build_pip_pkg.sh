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
# In addition, `tensorflow_compression/cc:libtensorflow_compression.so` should
# have been built using bazel.

set -ex  # Fail if any command fails, echo commands.

OUTPUT_DIR="${1-/tmp}"
WHEEL_VERSION="${2-0.dev0}"

PKGDIR="$(mktemp -d)"
trap 'rm -r -- "${PKGDIR}"' EXIT

cp LICENSE README.md MANIFEST.in requirements.txt build_pip_pkg.py "${PKGDIR}"

mkdir -p "${PKGDIR}/tensorflow_compression/cc"
cp "$(bazel info -c opt bazel-genfiles)/tensorflow_compression/cc/libtensorflow_compression.so" \
  "${PKGDIR}/tensorflow_compression/cc"


copy_file()
{
  FILENAME="${1#./}"
  DEST="${PKGDIR%/}/$(dirname "${FILENAME}")"
  mkdir -p "${DEST}"
  cp "${FILENAME}" "${DEST}"
}

copy_file "tensorflow_compression/__init__.py"
copy_file "tensorflow_compression/all_tests.py"

# Assumes no irregular characters in the filenames.
find tensorflow_compression/python -name "*.py" \
  | while read filename; do copy_file "${filename}"; done

pushd "${PKGDIR}"
python build_pip_pkg.py . "${OUTPUT_DIR}" "${WHEEL_VERSION}"
popd
