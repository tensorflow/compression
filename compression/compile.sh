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

# Consult https://www.tensorflow.org/versions/master/extend/adding_an_op
# for building custom op libraries for TensorFlow.
TF_CFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))') )
TF_LFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))') )
# If you get a link error during test, build with -D_GLIBCXX_USE_CXX11_ABI=0.
g++ -std=c++11 -shared kernels/pmf_to_cdf_op.cc kernels/range_coder.cc kernels/range_coder_ops.cc kernels/range_coder_ops_util.cc ops/coder_ops.cc -o _coder_ops.so -fPIC ${TF_CFLAGS[@]} ${TF_LFLAGS[@]} -O2 -undefined dynamic_lookup
