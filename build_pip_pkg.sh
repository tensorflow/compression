#!/usr/bin/env bash
# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
# This is based on
# https://github.com/tensorflow/custom-op/blob/master/build_pip_pkg.sh
# and modified for this project.
# ==============================================================================

set -e

# This script needs some improvements.
#   - Needs a flag to use 'python' or 'python3' to run setup.py.
#   - Needs a flag to control shared library file extension.
function main() {
  if [ -z "${1}" ]; then
    DEST=/tmp/tensorflow_compression
  else
    DEST="${1}"
  fi

  mkdir -p "${DEST}"
  DEST="$(readlink -f "${DEST}")"

  TMPDIR=$(mktemp -d -t tmp.XXXXXXXXXX)
  trap 'rm -rf "${TMPDIR}"' EXIT

  echo $(date) : "=== Using tmpdir: ${TMPDIR}"
  PKGDIR="${TMPDIR}/tensorflow_compression"

  echo $(date) : "=== Copying files"
  rsync -amqL tensorflow_compression/ "${PKGDIR}/"
  cp MANIFEST.in setup.py "${TMPDIR}"
  cp LICENSE README.md "${PKGDIR}"

  pushd ${TMPDIR} > /dev/null

  # Check if shared library file is copied. If this fails, it is likely that
  # this script was run directly from the workspace directory. This should be
  # run inside bazel-bin/build_pip_pkg.runfiles/tensorflow_compression/, or
  # simply run using "bazel run".
  if [ ! -f tensorflow_compression/cc/libtensorflow_compression.so ]; then
    echo "libtensorflow_compression.so not found. Did you use \"bazel run?\""
    exit 1
  fi

  echo $(date) : "=== Building wheel"
  python setup.py bdist_wheel

  echo $(date) : "=== Copying wheel to /tmp/"
  rsync -a --info=name dist/*.whl "${DEST}"
  popd > /dev/null

  echo $(date) : "=== Output wheel file is in: ${DEST}"
}

main "$@"
