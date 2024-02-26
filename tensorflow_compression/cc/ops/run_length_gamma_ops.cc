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

REGISTER_OP("RunLengthGammaEncode")
    .Input("data: int32")
    .Output("code: string")
    .SetShapeFn(shape_inference::ScalarShape)
    .Doc(R"doc(
Encodes `data` using run-length and Elias gamma coding.

data: An int32 tensor of values to be encoded.
code: An encoded scalar string.
)doc");

REGISTER_OP("RunLengthGammaDecode")
    .Input("code: string")
    .Input("shape: int32")
    .Output("data: int32")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle out;
      TF_RETURN_IF_ERROR(c->MakeShapeFromShapeTensor(1, &out));
      c->set_output(0, out);
      return absl::OkStatus();
    })
    .Doc(R"doc(
Decodes `data` using run-length and Elias gamma coding.

This is the inverse operation to `RunLengthGammaEncode`. The shape of the tensor
that was encoded must be known by the caller.

code: An encoded scalar string as returned by `RunLengthGammaEncode`.
shape: An int32 vector giving the shape of the encoded data.
data: An int32 tensor of decoded values, with shape `shape`.
)doc");

}  // namespace
}  // namespace tensorflow_compression
