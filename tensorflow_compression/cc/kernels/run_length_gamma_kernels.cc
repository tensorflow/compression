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
#define EIGEN_USE_THREADS

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <type_traits>
#include <vector>

#include "absl/types/span.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow_compression/cc/lib/bit_coder.h"

namespace tensorflow_compression {
namespace {
namespace errors = tensorflow::errors;
using tensorflow::DEVICE_CPU;
using tensorflow::OpKernel;
using tensorflow::OpKernelConstruction;
using tensorflow::OpKernelContext;
using tensorflow::string;
using tensorflow::Tensor;
using tensorflow::TensorShape;
using tensorflow::TensorShapeUtils;
using tensorflow::tstring;

#define OP_REQUIRES_OK_ABSL(context, status) \
  {                                                                    \
    auto s = (status);                                                 \
    OP_REQUIRES(context, s.ok(), tensorflow::Status(                   \
        static_cast<tensorflow::errors::Code>(s.code()), s.message())); \
  }

class RunLengthGammaEncodeOp : public OpKernel {
 public:
  explicit RunLengthGammaEncodeOp(OpKernelConstruction* context)
      : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    const Tensor& data_tensor = context->input(0);
    auto data = data_tensor.flat<int32_t>();

    Tensor* code_tensor;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, TensorShape{}, &code_tensor));
    tstring* code = &code_tensor->scalar<tstring>()();

    BitWriter enc;
    // Save number of zeros + 1 preceding next non-zero element.
    uint32_t zero_ct = 1;

    // Iterate through data tensor.
    for (int64_t i = 0; i < data.size(); i++) {
      int32_t sample = data(i);
      // Increment zero count.
      if (sample == 0) {
        zero_ct += 1;
      } else {
        // Encode run length of zeros.
        enc.WriteGamma(zero_ct);
        // Encode sign of value.
        enc.WriteOneBit(sample > 0);
        // Encode magnitude of value.
        if (sample == std::numeric_limits<int32_t>::min()) {
          // We can't encode int32 minimum. Encode closest value instead.
          sample += 1;
        }
        enc.WriteGamma(std::abs(sample));
        // Reset zero count (1 because gamma cannot encode 0).
        zero_ct = 1;
      }
    }
    if (zero_ct > 1) {
      enc.WriteGamma(zero_ct);
    }

    // Write encoded bitstring to code.
    auto encoded = enc.GetData();
    code->assign(encoded.data(), encoded.size());
  }
};

REGISTER_KERNEL_BUILDER(Name("RunLengthGammaEncode").Device(DEVICE_CPU),
                        RunLengthGammaEncodeOp);

class RunLengthGammaDecodeOp : public OpKernel {
 public:
  explicit RunLengthGammaDecodeOp(OpKernelConstruction* context)
      : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    const Tensor& code_tensor = context->input(0);
    const Tensor& shape_tensor = context->input(1);

    OP_REQUIRES(
        context, TensorShapeUtils::IsScalar(code_tensor.shape()),
        errors::InvalidArgument("Invalid `code` shape: ", code_tensor.shape()));
    OP_REQUIRES(context, TensorShapeUtils::IsVector(shape_tensor.shape()),
                errors::InvalidArgument("Invalid `shape` shape: ",
                                        shape_tensor.shape()));

    const tstring& code = code_tensor.scalar<tstring>()();

    TensorShape data_shape;
    OP_REQUIRES_OK(context, TensorShapeUtils::MakeShape(
                                shape_tensor.vec<int32_t>(), &data_shape));
    Tensor* data_tensor;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, data_shape, &data_tensor));
    auto data = data_tensor->flat<int32_t>();

    // Initialize bit decoder to point at the code and expect code size bytes.
    BitReader dec(code);

    // Fill data tensor with zeros.
    std::memset(data.data(), 0, data.size() * sizeof(data(0)));

    for (int64_t i = 0; i < data.size(); i++) {
      // Get number of zeros.
      auto num_zeros = dec.ReadGamma();
      OP_REQUIRES_OK_ABSL(context, num_zeros.status());

      // Advance the index to the next non-zero element.
      i += *num_zeros - 1;

      // Account for case where the last element is zero.
      // Check if past the last element.
      if (i >= data.size()) {
        OP_REQUIRES(context, i == data.size(),
                    errors::DataLoss("Decoded past end of tensor."));
        break;
      }

      // Get sign of value.
      auto positive = dec.ReadOneBit();
      OP_REQUIRES_OK_ABSL(context, positive.status());

      // Get magnitude.
      auto magnitude = dec.ReadGamma();
      OP_REQUIRES_OK_ABSL(context, magnitude.status());

      // Write value to data tensor element at index.
      data(i) = *positive ? *magnitude : -*magnitude;
    }
  }
};

REGISTER_KERNEL_BUILDER(Name("RunLengthGammaDecode").Device(DEVICE_CPU),
                        RunLengthGammaDecodeOp);

#undef OP_REQUIRES_OK_ABSL

}  // namespace
}  // namespace tensorflow_compression
