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
using tensorflow::Status;
using tensorflow::shape_inference::DimensionHandle;
using tensorflow::shape_inference::InferenceContext;
using tensorflow::shape_inference::ShapeHandle;

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
Converts a PMF into a quantized CDF for range coding.

This op uses floating-point operations internally. Therefore the quantized
output may not be consistent across multiple platforms. For entropy encoders and
decoders to have the same quantized CDF on different platforms, the quantized
CDF should be produced once and saved, then the saved quantized CDF should be
used everywhere.

After quantization, if PMF does not sum to 2^precision, then some values of PMF
are increased or decreased to adjust the sum to equal to 2^precision.

Note that the input PMF is pre-quantization. The input PMF is not normalized
by this op prior to quantization. Therefore the user is responsible for
normalizing PMF if necessary.
)doc");

}  // namespace
}  // namespace tensorflow_compression
