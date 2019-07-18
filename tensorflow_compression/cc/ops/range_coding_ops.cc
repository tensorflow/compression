/* Copyright 2018 Google LLC. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow_compression {
namespace {
namespace shape_inference = tensorflow::shape_inference;
using tensorflow::Status;
using tensorflow::shape_inference::DimensionHandle;
using tensorflow::shape_inference::InferenceContext;
using tensorflow::shape_inference::ShapeHandle;

REGISTER_OP("RangeEncode")
    .Input("data: int16")
    .Input("cdf: int32")
    .Output("encoded: string")
    .Attr("precision: int >= 1")
    .Attr("debug_level: int = 1")
    .SetShapeFn(shape_inference::ScalarShape)
    .Doc(R"doc(
Using the provided cumulative distribution functions (CDF) inside `cdf`, returns
a range-code of `data`.

The shape of `cdf` should have one more axis than the shape of `data`, and the
prefix `cdf.shape[:-1]` should be broadcastable to `data.shape`. That is, for
every `i = 0,...,rank(data) - 1`, the op requires that either
`cdf.shape[i] == 1` or `cdf.shape[i] == data.shape[i]`. Note that this
broadcasting is limited in the sense that the number of axes must match, and
broadcasts only `cdf` but not `data`.

`data` should have an upper bound `m > 0` such that each element is an integer
in range `[0, m)`. Then the last dimension size of `cdf` must be `m + 1`. For
each element of `data`, the innermost strip of `cdf` is a vector representing a
CDF. For each k = 0,...,m, `cdf[..., k] / 2^precision` is the probability that
an outcome is less than `k` (not less than or equal to).

```
   cdf[..., 0] / 2^precision = Pr(data[...] < 0)
   cdf[..., 1] / 2^precision = Pr(data[...] < 1) = Pr(data[...] <= 0)
   cdf[..., 2] / 2^precision = Pr(data[...] < 2) = Pr(data[...] <= 1)
   ...
   cdf[..., m] / 2^precision = Pr(data[...] < m) = 1
```

Therefore each element of `cdf` must be in `[0, 2^precision]`.

Ideally `cdf[..., m]` should equal to `2^precision` but this is not a hard
requirement as long as `cdf[..., m] <= 2^precision`.

The encoded string neither contains the shape information of the encoded data
nor a termination symbol. Therefore the shape of the encoded data must be
explicitly provided to the decoder.

Implementation notes:

- Because of potential performance issues, the op does not check whether
elements of `data` is in the correct range `[0, m)`, or if `cdf` satisfies
monotonic increase property.

- For the range coder to decode the encoded string correctly, the decoder should
be able to reproduce the internal states of the encoder precisely. Otherwise,
the decoding would fail and once an error occur, all subsequent decoded values
are incorrect. For this reason, the range coder uses integer arithmetics and
avoids using any floating point operations internally, and `cdf` should contain
integers representing quantized probability mass rather than floating points.

data: An int16 tensor.
cdf: An int32 tensor representing the CDF's of `data`. Each integer is divided
  by `2^precision` to represent a fraction.
encoded: A range-coded scalar string.
precision: The number of bits for probability quantization. Must be <= 16.
debug_level: Either 0 or 1.
)doc");

REGISTER_OP("RangeDecode")
    .Input("encoded: string")
    .Input("shape: int32")
    .Input("cdf: int32")
    .Output("decoded: int16")
    .Attr("precision: int >= 1")
    .Attr("debug_level: int = 1")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle out;
      TF_RETURN_IF_ERROR(c->MakeShapeFromShapeTensor(1, &out));
      c->set_output(0, out);
      return Status::OK();
    })
    .Doc(R"doc(
Decodes a range-coded `code` into an int32 tensor of shape `shape`.

This is the reverse op of RangeEncode. The shape of the tensor that was encoded
should be known by the caller.

Implementation notes:

- If wrong input was given (e.g., corrupt `encoded` string, or `cdf` or
`precision` do not match encoder), the decode is unsuccessful. Because of
potential performance issues, the decoder does not return error status.

encoded: A scalar string tensor from RangeEncode.
shape: An int32 1-D tensor representing the shape of the data encoded by
  RangeEncode.
decoded: An int16 tensor with shape equal to `shape`.
precision: The number of bits for probability quantization. Must be <= 16, and
  must match the precision used by RangeEncode that produced `encoded`.
debug_level: Either 0 or 1.
)doc");

REGISTER_OP("UnboundedIndexRangeEncode")
    .Input("data: int32")
    .Input("index: int32")
    .Input("cdf: int32")
    .Input("cdf_size: int32")
    .Input("offset: int32")
    .Output("encoded: string")
    .Attr("precision: int >= 1")
    .Attr("overflow_width: int >= 1")
    .Attr("debug_level: int = 1")
    .SetShapeFn(shape_inference::ScalarShape)
    .Doc(R"doc(
Range encodes unbounded integer `data` using an indexed probability table.

For each value in `data`, the corresponding value in `index` determines which
probability model in `cdf` is used to encode it. The data can be arbitrary
signed integers, where the integer intervals determined by `offset` and
`cdf_size` are modeled using the cumulative distribution functions (CDF) in
`cdf`. Everything else is encoded with a variable length code.

The argument `cdf` is a 2-D tensor and its each row contains a CDF. The argument
`cdf_size` is a 1-D tensor, and its length should be the same as the number of
rows of `cdf`. The values in `cdf_size` denotes the length of CDF vector in the
corresponding row of `cdf`.

For i = 0,1,..., let `m = cdf_size[i]`. Then for j = 0,1,...,m-1,

```
   cdf[..., 0] / 2^precision = Pr(X < 0) = 0
   cdf[..., 1] / 2^precision = Pr(X < 1) = Pr(X <= 0)
   cdf[..., 2] / 2^precision = Pr(X < 2) = Pr(X <= 1)
   ...
   cdf[..., m-1] / 2^precision = Pr(X < m-1) = Pr(X <= m-2).
```

We require that `1 < m <= cdf.shape[1]` and that all elements of `cdf` be in the
closed interval `[0, 2^precision]`.

Arguments `data` and `index` should have the same shape. `data` contains the
values to be encoded. `index` denotes which row in `cdf` should be used to
encode the corresponding value in `data`, and which element in `offset`
determines the integer interval the cdf applies to. Naturally, the elements of
`index` should be in the half-open interval `[0, cdf.shape[0])`.

When a value from `data` is in the interval `[offset[i], offset[i] + m - 2)`,
then the value is range encoded using the CDF values. The last entry in each
CDF (the one at `m - 1`) is an overflow code. When a value from `data` is
outside of the given interval, the overflow value is encoded, followed by a
variable-length encoding of the actual data value.

The encoded output contains neither the shape information of the encoded data
nor a termination symbol. Therefore the shape of the encoded data must be
explicitly provided to the decoder.

Implementation notes:

- Because of potential performance issues, the op does not check if `cdf`
satisfies monotonic increase property.

- For the range coder to decode the encoded string correctly, the decoder should
be able to reproduce the internal states of the encoder precisely. Otherwise,
the decoding would fail and once an error occur, all subsequent decoded values
are incorrect. For this reason, the range coder uses integer arithmetics and
avoids using any floating point operations internally, and `cdf` should contain
integers representing quantized probability mass rather than floating points.

data: An int32 tensor.
index: An int32 tensor of the same shape as `data`.
cdf: An int32 tensor representing the CDF's of `data`. Each integer is divided
  by `2^precision` to represent a fraction.
cdf_size: An int32 tensor.
offset: An int32 tensor.
encoded: A range-coded scalar string and a prefix varint string.
precision: The number of bits for probability quantization. Must be <= 16.
overflow_width: The bit width of the variable-length overflow code. Must be <=
  precision.
)doc");

REGISTER_OP("UnboundedIndexRangeDecode")
    .Input("encoded: string")
    .Input("index: int32")
    .Input("cdf: int32")
    .Input("cdf_size: int32")
    .Input("offset: int32")
    .Output("decoded: int32")
    .Attr("precision: int >= 1")
    .Attr("overflow_width: int >= 1")
    .Attr("debug_level: int = 1")
    .SetShapeFn([](InferenceContext* c) {
      c->set_output(0, c->input(1));
      return Status::OK();
    })
    .Doc(R"doc(
This is the reverse op of `UnboundedIndexRangeEncode`, and decodes the range
encoded stream `code` into an int32 tensor `decoded`. The other inputs `index`,
`cdf`, `cdf_size`, and `offset` should be the identical tensors passed to the
`UnboundedIndexRangeEncode` op that generated the `decoded` tensor.

Implementation notes:

- If a wrong input was given (e.g., a corrupt `encoded` string, or `cdf` or
`precision` not matching the encoder), the decode is unsuccessful. Because of
potential performance issues, the decoder does not return an error status.

encoded: A scalar string tensor from `UnboundedIndexRangeEncode`.
index: An int32 tensor of the same shape as `data`.
cdf: An int32 tensor representing the CDF's of `data`. Each integer is divided
  by `2^precision` to represent a fraction.
cdf_size: An int32 tensor.
offset: An int32 tensor.
decoded: An int32 tensor with the same shape as `index`.
precision: The number of bits for probability quantization. Must be <= 16, and
  must match the precision used by `UnboundedIndexRangeEncode` that produced
  `encoded`.
overflow_width: The bit width of the variable-length overflow code. Must be <=
  precision, and must match the width used by `UnboundedIndexRangeEncode` that
  produced `encoded`.
)doc");

REGISTER_OP("PmfToQuantizedCdf")
    .Input("pmf: float")
    .Output("cdf: int32")
    .Attr("precision: int >= 1")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle in;
      TF_RETURN_IF_ERROR(c->WithRankAtLeast(c->input(0), 1, &in));
      DimensionHandle last;
      TF_RETURN_IF_ERROR(c->Add(c->Dim(in, -1), 1, &last));
      ShapeHandle out;
      TF_RETURN_IF_ERROR(c->ReplaceDim(in, -1, last, &out));
      c->set_output(0, out);
      return Status::OK();
    })
    .Doc(R"doc(
Converts PMF to quantized CDF. This op uses floating-point operations
internally. Therefore the quantized output may not be consistent across multiple
platforms. For entropy encoders and decoders to have the same quantized CDF on
different platforms, the quantized CDF should be produced once and saved, then
the saved quantized CDF should be used everywhere.

After quantization, if PMF does not sum to 2^precision, then some values of PMF
are increased or decreased to adjust the sum to equal to 2^precision.

Note that the input PMF is pre-quantization. The input PMF is not normalized
by this op prior to quantization. Therefore the user is responsible for
normalizing PMF if necessary.
)doc");

}  // namespace
}  // namespace tensorflow_compression
