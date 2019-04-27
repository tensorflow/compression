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
"""Padding ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


__all__ = [
    "same_padding_for_kernel",
]


def same_padding_for_kernel(shape, corr, strides_up=None):
  """Determine correct amount of padding for `same` convolution.

  To implement `'same'` convolutions, we first pad the image, and then perform a
  `'valid'` convolution or correlation. Given the kernel shape, this function
  determines the correct amount of padding so that the output of the convolution
  or correlation is the same size as the pre-padded input.

  Args:
    shape: Shape of the convolution kernel (without the channel dimensions).
    corr: Boolean. If `True`, assume cross correlation, if `False`, convolution.
    strides_up: If this is used for an upsampled convolution, specify the
      strides here. (For downsampled convolutions, specify `(1, 1)`: in that
      case, the strides don't matter.)

  Returns:
    The amount of padding at the beginning and end for each dimension.
  """
  rank = len(shape)
  if strides_up is None:
    strides_up = rank * (1,)

  if corr:
    padding = [(s // 2, (s - 1) // 2) for s in shape]
  else:
    padding = [((s - 1) // 2, s // 2) for s in shape]

  padding = [((padding[i][0] - 1) // strides_up[i] + 1,
              (padding[i][1] - 1) // strides_up[i] + 1) for i in range(rank)]
  return padding
