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

#define OP_REQUIRES_OK_ABSL(context, status)                                \
  {                                                                         \
    auto s = (status);                                                      \
    OP_REQUIRES(                                                            \
        context, s.ok(),                                                    \
        tensorflow::Status(static_cast<tensorflow::errors::Code>(s.code()), \
                           s.message()));                                   \
  }

// TODO(jonycgn): Try to avoid in-loop branches based on attributes.

class RunLengthEncodeOp : public OpKernel {
 public:
  explicit RunLengthEncodeOp(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context,
                   context->GetAttr("run_length_code", &run_length_code_));
    OP_REQUIRES_OK(context,
                   context->GetAttr("magnitude_code", &magnitude_code_));
    OP_REQUIRES_OK(context,
                   context->GetAttr("use_run_length_for_non_zeros",
                                    &use_run_length_for_non_zeros_));
  }

  inline void WriteRunLength(BitWriter& enc, const int32_t run_length) {
    if (run_length_code_ >= 0) {
      enc.WriteRice(run_length, run_length_code_);
    } else {
      enc.WriteGamma(run_length + 1);
    }
  }

  inline void WriteNonZero(BitWriter& enc, const int32_t sample) {
    assert(sample != 0);
    const int32_t sign = sample > 0;
    enc.WriteOneBit(sign);
    if (magnitude_code_ >= 0) {
      enc.WriteRice(sign ? sample - 1 : -(sample + 1),
                    magnitude_code_);
    } else {
      if (sample == std::numeric_limits<int32_t>::min()) {
        // We can't encode int32 minimum. Encode closest value instead.
        enc.WriteGamma(-(std::numeric_limits<int32_t>::min() + 1));
      } else {
        enc.WriteGamma(sign ? sample : -sample);
      }
    }
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& data_tensor = context->input(0);
    auto data = data_tensor.flat<int32_t>();

    Tensor* code_tensor;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, TensorShape{}, &code_tensor));
    tstring* code = &code_tensor->scalar<tstring>()();

    BitWriter enc;

    const int32_t* const end = data.data() + data.size();
    const int32_t* p = data.data();

    // If we encode both zeros and non-zeros with run-length encoding
    // (use_run_length_for_non_zeros == true), only the first zero run length
    // can possibly be zero. We can subtract 1 from all subsequent run lengths.
    int32_t run_length_offset = 0;

    while (p < end) {
      // Find next non-zero.
      const int32_t* q = std::find_if_not(p, end,
                                          [](int32_t x) { return x == 0; });
      WriteRunLength(enc, q - p - run_length_offset);
      p = q;

      if (!(p < end)) break;

      if (use_run_length_for_non_zeros_) {
        // Find next zero.
        q = std::find_if(p, end, [](int32_t x) { return x == 0; });
        WriteRunLength(enc, q - p - 1);
        while (p < q) {
          WriteNonZero(enc, *p++);
        }
        run_length_offset = 1;
      } else {
        WriteNonZero(enc, *p++);
      }
    }

    // Write encoded bitstring to code.
    auto encoded = enc.GetData();
    code->assign(encoded.data(), encoded.size());
  }

 private:
  int run_length_code_;
  int magnitude_code_;
  bool use_run_length_for_non_zeros_;
};

REGISTER_KERNEL_BUILDER(Name("RunLengthEncode").Device(DEVICE_CPU),
                        RunLengthEncodeOp);

class RunLengthDecodeOp : public OpKernel {
 public:
  explicit RunLengthDecodeOp(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context,
                   context->GetAttr("run_length_code", &run_length_code_));
    OP_REQUIRES_OK(context,
                   context->GetAttr("magnitude_code", &magnitude_code_));
    OP_REQUIRES_OK(context,
                   context->GetAttr("use_run_length_for_non_zeros",
                                    &use_run_length_for_non_zeros_));
  }

  inline absl::StatusOr<int32_t> ReadRunLength(OpKernelContext* context,
                                               BitReader& dec) {
    if (run_length_code_ >= 0) {
      return dec.ReadRice(run_length_code_);
    } else {
      auto gamma = dec.ReadGamma();
      if (!gamma.ok()) return gamma;
      return *gamma - 1;
    }
  }

  inline absl::StatusOr<int32_t> ReadNonZero(OpKernelContext* context,
                                             BitReader& dec) {
    auto positive = dec.ReadOneBit();
    if (!positive.ok()) return positive;
    if (magnitude_code_ >= 0) {
      auto rice = dec.ReadRice(magnitude_code_);
      if (!rice.ok()) return rice;
      return *positive ? *rice + 1 : -*rice - 1;
    } else {
      auto gamma = dec.ReadGamma();
      if (!gamma.ok()) return gamma;
      return *positive ? *gamma : -*gamma;
    }
  }

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

    // Initialize bit decoder to point at the code.
    BitReader dec(code);

    // Fill data tensor with zeros.
    std::memset(data.data(), 0, data.size() * sizeof(data(0)));

    int32_t* const end = data.data() + data.size();
    int32_t* p = data.data();

    // If we encode both zeros and non-zeros with run-length encoding
    // (use_run_length_for_non_zeros == true), only the first zero run length
    // can possibly be zero. We can subtract 1 from all subsequent run lengths.
    int32_t run_length_offset = 0;

    while (p < end) {
      // Skip to the next non-zero element.
      auto run_length = ReadRunLength(context, dec);
      OP_REQUIRES_OK_ABSL(context, run_length.status());

      p += *run_length + run_length_offset;

      if (!(p < end)) {
        // Should not be past the last element.
        OP_REQUIRES(context, p == end,
                    errors::DataLoss("Decoded past end of tensor."));
        break;
      }

      if (use_run_length_for_non_zeros_) {
        run_length = ReadRunLength(context, dec);
        OP_REQUIRES_OK_ABSL(context, run_length.status());
        const int32_t* const next_zero = p + *run_length + 1;
        OP_REQUIRES(context, next_zero <= end,
                    errors::DataLoss("Decoded past end of tensor."));
        while (p < next_zero) {
          auto nonzero = ReadNonZero(context, dec);
          OP_REQUIRES_OK_ABSL(context, nonzero.status());
          *p++ = *nonzero;
        }
        run_length_offset = 1;
      } else {
        auto nonzero = ReadNonZero(context, dec);
        OP_REQUIRES_OK_ABSL(context, nonzero.status());
        *p++ = *nonzero;
      }
    }
  }

 private:
  int run_length_code_;
  int magnitude_code_;
  bool use_run_length_for_non_zeros_;
};

REGISTER_KERNEL_BUILDER(Name("RunLengthDecode").Device(DEVICE_CPU),
                        RunLengthDecodeOp);

#undef OP_REQUIRES_OK_ABSL

}  // namespace
}  // namespace tensorflow_compression
