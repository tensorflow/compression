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
"""Setup for pip package."""

import atexit
import glob
import os
import shutil
import sys
import tempfile
import setuptools

# Version string should follow PEP440 rules.
DEFAULT_VERSION = "0.dev0+build-from-source"


class BinaryDistribution(setuptools.Distribution):
  """This class is needed in order to create OS specific wheels."""

  def has_ext_modules(self):
    return True


def main(srcdir: str, destdir: str, version: str = ""):
  tempdir = tempfile.mkdtemp()
  atexit.register(shutil.rmtree, tempdir)

  pkgdir = os.path.join(tempdir, "tensorflow_compression")
  shutil.copytree(os.path.join(srcdir, "tensorflow_compression"), pkgdir)
  shutil.copy2(os.path.join(srcdir, "MANIFEST.in"), tempdir)
  shutil.copy2(os.path.join(srcdir, "LICENSE"), pkgdir)
  shutil.copy2(os.path.join(srcdir, "README.md"), pkgdir)

  if not os.path.exists(
      os.path.join(pkgdir, "cc/libtensorflow_compression.so")):
    raise RuntimeError("libtensorflow_compression.so not found. "
                       "Did you 'bazel run?'")

  with open(os.path.join(srcdir, "requirements.txt"), "r") as f:
    install_requires = f.readlines()

  print("=== Building wheel")
  atexit.register(os.chdir, os.getcwd())
  os.chdir(tempdir)
  setuptools.setup(
      name="tensorflow_compression",
      version=version or DEFAULT_VERSION,
      description="Data compression in TensorFlow",
      url="https://tensorflow.github.io/compression/",
      author="Google LLC",
      # Contained modules and scripts.
      packages=setuptools.find_packages(),
      install_requires=install_requires,
      script_args=["sdist", "bdist_wheel"],
      # Add in any packaged data.
      include_package_data=True,
      zip_safe=False,
      distclass=BinaryDistribution,
      # PyPI package information.
      classifiers=[
          "Development Status :: 5 - Production/Stable",
          "Intended Audience :: Developers",
          "Intended Audience :: Education",
          "Intended Audience :: Science/Research",
          "License :: OSI Approved :: Apache Software License",
          "Programming Language :: Python :: 3",
          "Topic :: Scientific/Engineering :: Mathematics",
          "Topic :: Software Development :: Libraries :: Python Modules",
          "Topic :: Software Development :: Libraries",
      ],
      project_urls={
          "Documentation":
              "https://tensorflow.github.io/compression/docs/api_docs/python/tfc.html",
          "Discussion":
              "https://groups.google.com/forum/#!forum/tensorflow-compression",
          "Source": "https://github.com/tensorflow/compression",
          "Tracker": "https://github.com/tensorflow/compression/issues",
      },
      license="Apache 2.0",
      keywords=("compression data-compression tensorflow machine-learning "
                "python deep-learning deep-neural-networks neural-network ml")
  )

  print("=== Copying wheel to:", destdir)
  os.makedirs(destdir, exist_ok=True)
  for path in glob.glob(os.path.join(tempdir, "dist", "*.whl")):
    print("Copied into:", shutil.copy(path, destdir))


if __name__ == "__main__":
  main(*sys.argv[1:])  # pylint: disable=too-many-function-args

