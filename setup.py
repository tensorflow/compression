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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from setuptools import find_packages
from setuptools import setup
from setuptools.dist import Distribution

# Version string is intentionally set to non-numeric value, so that non-release
# built packages are different from release packages. During builds for formal
# releases, we should temporarily change this value to pip release version.
__version__ = 'custom-build-from-source'


class BinaryDistribution(Distribution):
  """This class is needed in order to create OS specific wheels."""

  def has_ext_modules(self):
    return True


setup(
    name='tensorflow-compression',
    version=__version__,
    description=('Data compression in TensorFlow'),
    url='https://tensorflow.github.io/compression/',
    author='Google LLC',
    # Contained modules and scripts.
    packages=find_packages(),
    install_requires=[
        'scipy >= 1.0.0',
        'tensorflow >= 1.14.0',
    ],
    # Add in any packaged data.
    include_package_data=True,
    zip_safe=False,
    distclass=BinaryDistribution,
    # PyPI package information.
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.4',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: Software Development :: Libraries',
    ],
    project_urls={
        'Documentation': 'https://tensorflow.github.io/compression/docs/api_docs/python/tfc.html',
        'Discussion': 'https://groups.google.com/forum/#!forum/tensorflow-compression',
        'Source': 'https://github.com/tensorflow/compression',
        'Tracker': 'https://github.com/tensorflow/compression/issues',
    },
    license='Apache 2.0',
    keywords='compression data-compression tensorflow machine-learning python deep-learning deep-neural-networks neural-network ml',
)
