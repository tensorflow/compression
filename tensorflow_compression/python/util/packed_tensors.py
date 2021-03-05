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
"""Packed tensors in bit sequences."""

import tensorflow as tf


__all__ = [
    "PackedTensors",
]


class PackedTensors:
  """Packed representation of compressed tensors.

  This class can pack and unpack several tensor values into a single string. It
  can also optionally store a model identifier.

  The tensors currently must be rank 1 (vectors) and either have integer or
  string type.
  """

  def __init__(self, string=None):
    self._example = tf.train.Example()
    if string:
      self.string = string

  @property
  def model(self):
    """A model identifier."""
    buf = self._example.features.feature["MD"].bytes_list.value[0]
    return buf.decode("ascii")

  @model.setter
  def model(self, value):
    self._example.features.feature["MD"].bytes_list.value[:] = [
        value.encode("ascii")]

  @model.deleter
  def model(self):
    del self._example.features.feature["MD"]

  @property
  def string(self):
    """The string representation of this object."""
    return self._example.SerializeToString()

  @string.setter
  def string(self, value):
    self._example.ParseFromString(value)

  def pack(self, tensors):
    """Packs `Tensor` values into this object."""
    i = 1
    for tensor in tensors:
      feature = self._example.features.feature[chr(i)]
      feature.Clear()
      if tensor.shape.rank != 1:
        raise RuntimeError(f"Unexpected tensor rank: {tensor.shape.rank}.")
      if tensor.dtype.is_integer:
        feature.int64_list.value[:] = tensor.numpy()
      elif tensor.dtype == tf.string:
        feature.bytes_list.value[:] = tensor.numpy()
      else:
        raise RuntimeError(f"Unexpected tensor dtype: '{tensor.dtype}'.")
      i += 1
    # Delete any remaining, previously set arrays.
    while chr(i) in self._example.features.feature:
      del self._example.features.feature[chr(i)]
      i += 1

  def unpack(self, dtypes):
    """Unpacks values from this object based on dtypes."""
    tensors = []
    for i, dtype in enumerate(dtypes):
      dtype = tf.as_dtype(dtype)
      feature = self._example.features.feature[chr(i + 1)]
      if dtype.is_integer:
        tensors.append(tf.constant(feature.int64_list.value, dtype=dtype))
      elif dtype == tf.string:
        tensors.append(tf.constant(feature.bytes_list.value, dtype=dtype))
      else:
        raise RuntimeError(f"Unexpected dtype: '{dtype}'.")
    return tensors
