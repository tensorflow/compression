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

namespace tensorflow_compression {
namespace {

REGISTER_OP("StochasticRound")
    .Attr("T: {bfloat16, float16, float32}")
    .Input("inputs: T")
    .Input("step_size: float32")
    .Input("seed: int32")
    .Output("outputs: int32")
    .SetShapeFn(tensorflow::shape_inference::UnchangedShape)
    .Doc(R"doc(
Rounds `inputs / step_size` stochastically.

This op computes the elementwise function:

output = {
  floor(x)       with prob.   p = x - floor(x)
  floor(x) + 1   with prob.   1 - p
}
where x = input / step_size.

inputs: Floating point tensor to be rounded.
step_size: Scalar tensor. Step size for rounding.
seed: Arbitrary shape tensor. Seed for random number generator. If it has no
  elements, seeding is attempted from system time.
outputs: Integer tensor of same shape as `inputs`, containing rounded values.
)doc");

}  // namespace
}  // namespace tensorflow_compression
