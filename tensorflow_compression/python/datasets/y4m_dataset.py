# Copyright 2021 Google LLC. All Rights Reserved.
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
"""Y4M dataset."""

import tensorflow as tf
from tensorflow_compression.python.ops import gen_ops


__all__ = [
    "Y4MDataset",
]


class Y4MDataset(tf.data.Dataset):
  """A `tf.Dataset` of Y'CbCr video frames from '.y4m' files.

  This dataset yields tuples of `tf.uint8` tensors, where each tuple represents
  one video frame. It reads all files sequentially, and concatenates all frames
  into one big linear sequence.

  The first tensor contains the luma plane (Y') and has shape `(H, W, 1)`, where
  `H` and `W` are the height and width of the frame, respectively. The second
  tensor contains the two chroma planes (CbCr) and has shape `(Hc, Wc, 2)`.
  If the file uses 4:2:0 chroma format with vertically and horizontally
  interstitially sited chroma pixels (a.k.a. JPEG or MPEG1-style chroma
  alignment, marked in the file as `C420jpeg`), then `Hc == H/2` and
  `Wc == W/2`. If the file uses 4:4:4 chroma format (marked in the file as
  `C444`), then `Hc == H` and `Wc == W`.

  Other chroma formats (as well as interlaced frame formats) are currently not
  supported. Note that this means that the dataset refuses to read files with
  other 4:2:0 chroma alignments (for example, DV or MPEG-2 styles). Any other
  markers in the file (such as frame rate, pixel aspect ratio etc.) are
  silently ignored.
  """

  def __init__(self, filenames):
    """Creates a `Y4MDataset`.

    Args:
      filenames: A `tf.string` tensor containing one or more filenames.
    """
    super().__init__(gen_ops.y4m_dataset(filenames))

  def _inputs(self):
    return []

  @property
  def element_spec(self):
    return (tf.TensorSpec([None, None, 1], tf.uint8),
            tf.TensorSpec([None, None, 2], tf.uint8))
