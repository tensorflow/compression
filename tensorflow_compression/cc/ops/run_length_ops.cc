/* Copyright 2022 Google LLC. All Rights Reserved.

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
using tensorflow::shape_inference::InferenceContext;
using tensorflow::shape_inference::ShapeHandle;

REGISTER_OP("RunLengthEncode")
    .Attr("run_length_code: int")
    .Attr("magnitude_code: int")
    .Attr("use_run_length_for_non_zeros: bool")
    .Input("data: int32")
    .Output("code: string")
    .SetShapeFn(shape_inference::ScalarShape)
    .Doc(R"doc(
Encodes `data` using run-length coding.

This op implements a superset of RunLengthGammaEncode, which is equivalent to
calling RunLengthEncode with run_length_code = -1, magnitude_code = -1, and
use_run_length_for_non_zeros = false.

run_length_code: If >= 0, use Rice code with this parameter to encode run
  lengths, else use Golomb code.
magnitude_code: If >= 0, use Rice code with this parameter to encode magnitudes,
  else use Golomb code.
use_run_length_for_non_zeros: If true, alternate between coding run lengths of
  zeros and non-zeros. If false, only encode run lengths of zeros, and encode
  non-zeros one by one.
data: An int32 tensor of values to be encoded.
code: An encoded scalar string.
)doc");

REGISTER_OP("RunLengthDecode")
    .Attr("run_length_code: int")
    .Attr("magnitude_code: int")
    .Attr("use_run_length_for_non_zeros: bool")
    .Input("code: string")
    .Input("shape: int32")
    .Output("data: int32")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle out;
      TF_RETURN_IF_ERROR(c->MakeShapeFromShapeTensor(1, &out));
      c->set_output(0, out);
      return tensorflow::OkStatus();
    })
    .Doc(R"doc(
Decodes `data` using run-length coding.

This is the inverse operation to `RunLengthEncode`. The shape of the tensor
that was encoded must be known by the caller.

This op implements a superset of RunLengthGammaDecode, which is equivalent to
calling RunLengthDecode with run_length_code = -1, magnitude_code = -1, and
use_run_length_for_non_zeros = false.

run_length_code: If >= 0, use Rice code with this parameter to decode run
  lengths, else use Golomb code.
magnitude_code: If >= 0, use Rice code with this parameter to decode magnitudes,
  else use Golomb code.
use_run_length_for_non_zeros: If true, alternate between coding run lengths of
  zeros and non-zeros. If false, only decode run lengths of zeros, and decode
  non-zeros one by one.
code: An encoded scalar string as returned by `RunLengthEncode`.
shape: An int32 vector giving the shape of the encoded data.
data: An int32 tensor of decoded values, with shape `shape`.
)doc");

}  // namespace
}  // namespace tensorflow_compression
