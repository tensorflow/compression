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
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow_compression {
namespace {
namespace shape_inference = tensorflow::shape_inference;

REGISTER_OP("Y4MDataset")
    .Input("filenames: string")
    .Output("handle: variant")
    .SetDoNotOptimize()  // TODO(unassigned): Prevents constant folding.
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      shape_inference::ShapeHandle unused;
      TF_RETURN_IF_ERROR(c->WithRankAtMost(c->input(0), 1, &unused));
      return shape_inference::ScalarShape(c);
    })
    .Doc(R"doc(
Reads a sequence of .y4m files.

This op yields tuples of `tf.uint8` tensors, where each tuple represents one
video frame. It reads all files sequentially, and concatenates all frames into
one big linear sequence.

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
)doc");

}  // namespace
}  // namespace tensorflow_compression
