/* Copyright 2021 Google LLC. All Rights Reserved.

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
#include "tensorflow/core/framework/op_def_builder.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow_compression {

using tensorflow::Status;
using tensorflow::shape_inference::InferenceContext;
using tensorflow::shape_inference::ShapeHandle;

REGISTER_OP("CreateRangeEncoder")
    .Input("shape: int32")
    .Input("lookup: int32")
    .Output("handle: variant")
    .SetIsStateful()
    // RandomShape() makes the output shape from the input tensor.
    .SetShapeFn(tensorflow::shape_inference::RandomShape)
    .Doc(R"doc(
Creates range encoder objects to be used by `EntropyEncode*` ops.

The output `handle` has the shape specified by the input `shape`. Each element
in `handle` is an independent range encoder object, and `EntropyEncode*`
processes as many concurrent code streams as contained in `handle`.

This op expects `lookup` to be either a concatenation (1-D) or stack (2-D) of
CDFs, where each CDF is preceded by a corresponding precision value. In case of
a stack:

```
   lookup[..., 0] = precision in [1, 16],
   lookup[..., 1] / 2^precision = Pr(X < 0) = 0,
   lookup[..., 2] / 2^precision = Pr(X < 1),
   lookup[..., 3] / 2^precision = Pr(X < 2),
   ...
   lookup[..., -1] / 2^precision = 1,
```

Subsequent values in each CDF may be equal, indicating a symbol with zero
probability. Attempting to encode such a symbol will result in undefined
behavior. However, any number of trailing zero-probability symbols will be
interpreted as padding, and attempting to use those will result in an encoding
error (unless overflow functionality is used).

Overflow functionality can be enabled by negating the precision value in
`lookup`. In that case, the last non-zero probability symbol in the CDF is used
as an escape code, allowing negative integers and integers greater or equal to
the last non-zero probability symbol to be encoded using an Elias gamma code,
which is interleaved into the code stream. Attempting to encode a
zero-probability symbol within the valid range still causes undefined behavior.
)doc");

REGISTER_OP("EntropyEncodeChannel")
    .Input("handle: variant")
    .Input("value: Tvalue")
    .Output("aliased_handle: variant")
    .Attr("Tvalue : {int32}")
    .SetShapeFn(tensorflow::shape_inference::UnchangedShape)
    .Doc(R"doc(
Encodes each input in `value`.

In general, entropy encoders in `handle` reference multiple distributions.
The last (innermost) dimension of `value` determines which distribution is used
to encode `value`. For example, if `value` is a 3-D array, then `value(i,j,k)`
is encoded using the `k`-th distribution.

`handle` controls the number of coding streams. Suppose that `value` has the
shape `[2, 3, 4]` and that `handle` has shape `[2]`. Then the first slice
`[0, :, :]` of shape `[3, 4]` is encoded into `handle[0]` and the second
slice `[1, :, :]` is encoded into `handle[1]`. If `handle` has shape `[]`, then
there is only one handle, and the entire input is encoded into a single stream.

Values must be in the provided ranges specified when the input `handle` was
originally created, unless overflow functionality was enabled. The `handle` may
be produced by the `CreateRangeEncoder` op, or may be passed through from a
different `EntropyEncodeChannel/EntropyEncodeIndex` op.

Because the op modifies `handle`, the corresponding input edge to the op nodes
of this type should not have other consumers in the graph.
)doc");

REGISTER_OP("EntropyEncodeIndex")
    .Input("handle: variant")
    .Input("index: Tindex")
    .Input("value: Tvalue")
    .Output("aliased_handle: variant")
    .Attr("Tindex : {int32}")
    .Attr("Tvalue : {int32}")
    .SetShapeFn(tensorflow::shape_inference::UnchangedShape)
    .Doc(R"doc(
Encodes each input in `value` according to a distribution selected by `index`.

In general, entropy encoders in `handle` reference multiple distributions.
`index` selects which distribution is used to encode `value`. For example, if
`value` is a 3-D array, then `value(i,j,k)` is encoded using the
`index(i,j,k)`-th distribution. `index` and `value` must have the same shape.

`handle` controls the number of coding streams. Suppose that `value` and `index`
have the shape `[2, 3, 4]` and that `handle` has shape `[2]`. Then the first
slice `[0, :, :]` of shape `[3, 4]` is encoded into `handle[0]` and the second
slice `[1, :, :]` is encoded into `handle[1]`. If `handle` has shape `[]`, then
there is only one handle, and the entire input is encoded into a single stream.

Values must be in the provided ranges specified when the input `handle` was
originally created, unless overflow functionality was enabled. The `handle` may
be produced by the `CreateRangeEncoder` op, or may be passed through from a
different `EntropyEncodeChannel/EntropyEncodeIndex` op.

Because the op modifies `handle`, the corresponding input edge to the op nodes
of this type should not have other consumers in the graph.
)doc");

REGISTER_OP("EntropyEncodeFinalize")
    .Input("handle: variant")
    .Output("encoded: string")
    .SetShapeFn(tensorflow::shape_inference::UnchangedShape)
    .Doc(R"doc(
Finalizes the encoding process and extracts byte stream from the encoder.
)doc");

REGISTER_OP("CreateRangeDecoder")
    .Input("encoded: string")
    .Input("lookup: int32")
    .Output("handle: variant")
    .SetIsStateful()
    .SetShapeFn(tensorflow::shape_inference::UnchangedShape)
    .Doc(R"doc(
Creates range decoder objects to be used by `EntropyDecode*` ops.

The input `encoded` is referenced by `handle`. No op should modify the strings
contained in `encoded` while `handle` is alive.

encoded: A string tensor which contains the code stream. Typically produced by
  `EntropyEncodeFinalize`.
lookup: An int32 1-D or 2-D tensor. This should match the `lookup` argument of
  the corresponding `CreateRangeEncoder` op.
)doc");

REGISTER_OP("EntropyDecodeChannel")
    .Input("handle: variant")
    .Input("shape: int32")
    .Output("aliased_handle: variant")
    .Output("decoded: Tdecoded")
    .Attr("Tdecoded: {int32}")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle shape = c->input(0);
      c->set_output(0, shape);

      ShapeHandle suffix_shape;
      TF_RETURN_IF_ERROR(c->MakeShapeFromShapeTensor(1, &suffix_shape));
      TF_RETURN_IF_ERROR(c->Concatenate(shape, suffix_shape, &shape));
      c->set_output(1, shape);
      return Status::OK();
    })
    .Doc(R"doc(
Decodes the encoded stream inside `handle`.

The output shape is defined as `handle.shape + MakeShape(shape)`, and therefore
both `handle` and `shape` arguments determine how many symbols are decoded.

Like encoders, decoders in `handle` reference multiple distributions. The last
(innermost) dimension of `value` determines which distribution is used to decode
each value in the output. For example, if `decoded` is a 3-D array, then
`output(i,j,k)` is decoded using the `k`-th distribution.

`handle` controls the number of coding streams. Suppose that `index` has the
shape `[2, 3, 4]` and that `handle` has shape `[2]`. Then the first output slice
`[0, :, :]` of shape `[3, 4]` is decoded from `handle[0]` and the second output
slice `[1, :, :]` is decoded from `handle[1]`. If `handle` has shape `[]`, then
there is only one handle, and the entire output is decoded from a single stream.

The input handle may be produced by the `CreateRangeDecoder` op, or may be
passed through from a different `EntropyDecode*` op.

This op modifies the input `handle`. The handle input edge to the op nodes of
this type should not have other consumers in the graph.
)doc");

REGISTER_OP("EntropyDecodeIndex")
    .Input("handle: variant")
    .Input("index: Tindex")
    .Input("shape: int32")
    .Output("aliased_handle: variant")
    .Output("decoded: Tdecoded")
    .Attr("Tindex: {int32}")
    .Attr("Tdecoded: {int32}")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle shape = c->input(0);
      c->set_output(0, shape);

      ShapeHandle suffix_shape;
      TF_RETURN_IF_ERROR(c->MakeShapeFromShapeTensor(2, &suffix_shape));
      TF_RETURN_IF_ERROR(c->Concatenate(shape, suffix_shape, &shape));
      c->set_output(1, shape);
      return Status::OK();
    })
    .Doc(R"doc(
Decodes the encoded stream inside `handle`.

The output shape is defined as `handle.shape + MakeShape(shape)`, and therefore
both `handle` and `shape` arguments determine how many symbols are decoded.

Like encoders, decoders in `handle` reference multiple distributions. `index`
indicates which distribution should be used to decode each value in the output.
For example, if `decoded` is a 3-D array, then `output(i,j,k)` is decoded using
the `index(i,j,k)`-th distribution. In general, `index` should match the `index`
of the corresponding `EntropyEncodeIndex` op. `index` should have the same shape
as output `decoded`: `handle.shape + MakeShape(shape)`.

`handle` controls the number of coding streams. Suppose that `index` has the
shape `[2, 3, 4]` and that `handle` has shape `[2]`. Then the first output slice
`[0, :, :]` of shape `[3, 4]` is decoded from `handle[0]` and the second output
slice `[1, :, :]` is decoded from `handle[1]`. If `handle` has shape `[]`, then
there is only one handle, and the entire output is decoded from a single stream.

The input handle may be produced by the `CreateRangeDecoder` op, or may be
passed through from a different `EntropyDecode*` op.

This op modifies the input `handle`. The handle input edge to the op nodes of
this type should not have other consumers in the graph.
)doc");

REGISTER_OP("EntropyDecodeFinalize")
    .Input("handle: variant")
    .Output("success: bool")
    .SetShapeFn(tensorflow::shape_inference::UnchangedShape)
    .Doc(R"doc(
Finalizes the decoding process. This op performs a *weak* sanity check, and the
return value may be false if some catastrophic error has happened. This is a
quite weak safety device, and one should not rely on this for error detection.
)doc");

}  // namespace tensorflow_compression
