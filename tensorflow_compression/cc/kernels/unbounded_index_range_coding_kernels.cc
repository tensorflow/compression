/* Copyright 2019 Google LLC. All Rights Reserved.

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

// DEPRECATED. Use new implementation of range coders in range_coder_kernels.cc.

#include <algorithm>
#include <array>
#include <limits>
#include <numeric>
#include <type_traits>
#include <vector>

#define EIGEN_USE_THREADS

#include "absl/types/span.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow_compression/cc/lib/range_coder.h"

namespace tensorflow_compression {
namespace {
namespace errors = tensorflow::errors;
using tensorflow::DEVICE_CPU;
using tensorflow::int16;
using tensorflow::int32;
using tensorflow::int64;
using tensorflow::OpKernel;
using tensorflow::OpKernelConstruction;
using tensorflow::OpKernelContext;
using tensorflow::Status;
using tensorflow::string;
using tensorflow::Tensor;
using tensorflow::TensorShape;
using tensorflow::TensorShapeUtils;
using tensorflow::tstring;
using tensorflow::TTypes;
using tensorflow::uint32;
using tensorflow::uint64;
using tensorflow::uint8;

tensorflow::Status CheckIndex(int64 upper_bound, const Tensor& index) {
  auto flat = index.flat<int32>();
  for (int64 i = 0; i < flat.size(); ++i) {
    if (flat(i) < 0 || upper_bound <= flat(i)) {
      return errors::InvalidArgument("'index' has a value not in [0, ",
                                     upper_bound, "): value=", flat(i));
    }
  }
  return tensorflow::Status::OK();
}

tensorflow::Status CheckCdfSize(int64 upper_bound, const Tensor& cdf_size) {
  auto flat = cdf_size.vec<int32>();
  for (int64 i = 0; i < flat.size(); ++i) {
    if (flat(i) < 3 || upper_bound < flat(i)) {
      return errors::InvalidArgument("'cdf_size' has a value not in [3, ",
                                     upper_bound, "]: value=", flat(i));
    }
  }
  return tensorflow::Status::OK();
}

tensorflow::Status CheckCdf(int precision, const Tensor& cdf,
                            const Tensor& cdf_size) {
  auto matrix = cdf.matrix<int32>();
  auto size = cdf_size.vec<int32>();
  CHECK_EQ(matrix.dimension(0), size.size());
  CHECK_GT(matrix.dimension(1), 2);

  const int32 upper_bound = 1 << precision;

  for (int64 i = 0; i < matrix.dimension(0); ++i) {
    const TTypes<int32, 1>::ConstVec slice(&matrix(i, 0), size(i));
    if (slice(0) != 0 || slice(slice.size() - 1) != upper_bound) {
      return errors::InvalidArgument("Each cdf should start from 0 and end at ",
                                     upper_bound, ": cdf[0]=", slice(0),
                                     ", cdf[^1]=", slice(slice.size() - 1));
    }

    for (int64 j = 0; j + 1 < slice.size(); ++j) {
      if (slice(j + 1) <= slice(j)) {
        return errors::InvalidArgument("CDF is not monotonic");
      }
    }
  }
  return tensorflow::Status::OK();
}

// Assumes that CheckArgumentShapes().ok().
tensorflow::Status CheckArgumentValues(int precision, const Tensor& index,
                                       const Tensor& cdf,
                                       const Tensor& cdf_size,
                                       const Tensor& offset) {
  TF_RETURN_IF_ERROR(CheckIndex(cdf.dim_size(0), index));
  TF_RETURN_IF_ERROR(CheckCdfSize(cdf.dim_size(1), cdf_size));
  TF_RETURN_IF_ERROR(CheckCdf(precision, cdf, cdf_size));
  return tensorflow::Status::OK();
}

tensorflow::Status CheckArgumentShapes(const Tensor& index, const Tensor& cdf,
                                       const Tensor& cdf_size,
                                       const Tensor& offset) {
  if (!TensorShapeUtils::IsMatrix(cdf.shape()) || cdf.dim_size(1) < 3) {
    return errors::InvalidArgument(
        "'cdf' should be 2-D and cdf.dim_size(1) >= 3: ", cdf.shape());
  }
  if (!TensorShapeUtils::IsVector(cdf_size.shape()) ||
      cdf_size.dim_size(0) != cdf.dim_size(0)) {
    return errors::InvalidArgument(
        "'cdf_size' should be 1-D and its length "
        "should match the number of rows in 'cdf': ",
        cdf_size.shape());
  }
  if (!TensorShapeUtils::IsVector(offset.shape()) ||
      offset.dim_size(0) != cdf.dim_size(0)) {
    return errors::InvalidArgument(
        "'offset' should be 1-D and its length "
        "should match the number of rows in 'cdf': offset.shape=",
        offset.shape(), ", cdf.shape=", cdf.shape());
  }
  return tensorflow::Status::OK();
}

class UnboundedIndexRangeEncodeOp : public OpKernel {
 public:
  explicit UnboundedIndexRangeEncodeOp(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("precision", &precision_));
    OP_REQUIRES(context, 0 < precision_ && precision_ <= 16,
                errors::InvalidArgument("`precision` must be in [1, 16]: ",
                                        precision_));
    OP_REQUIRES_OK(context,
                   context->GetAttr("overflow_width", &overflow_width_));
    OP_REQUIRES(context, 0 < overflow_width_ && overflow_width_ <= 16,
                errors::InvalidArgument("`overflow_width` must be in [1, 16]: ",
                                        overflow_width_));
    OP_REQUIRES_OK(context, context->GetAttr("debug_level", &debug_level_));
    OP_REQUIRES(context, debug_level_ == 0 || debug_level_ == 1,
                errors::InvalidArgument("`debug_level` must be 0 or 1: ",
                                        debug_level_));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& data = context->input(0);
    const Tensor& index = context->input(1);
    const Tensor& cdf = context->input(2);
    const Tensor& cdf_size = context->input(3);
    const Tensor& offset = context->input(4);

    OP_REQUIRES(
        context, data.shape() == index.shape(),
        errors::InvalidArgument(
            "`data` and `index` should have the same shape: data.shape=",
            data.shape(), ", index.shape=", index.shape()));

    OP_REQUIRES_OK(context, CheckArgumentShapes(index, cdf, cdf_size, offset));
    if (debug_level_ > 0) {
      OP_REQUIRES_OK(context, CheckArgumentValues(precision_, index, cdf,
                                                  cdf_size, offset));
    }

    Tensor* output_tensor;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, TensorShape{}, &output_tensor));
    std::string output;

    RangeEncodeImpl(data.flat<int32>(), index.flat<int32>(),
                    cdf.matrix<int32>(), cdf_size.vec<int32>(),
                    offset.vec<int32>(), &output);
    output_tensor->flat<tstring>()(0) = output;
  }

 private:
  void RangeEncodeImpl(TTypes<int32>::ConstFlat data,
                       TTypes<int32>::ConstFlat index,
                       TTypes<int32>::ConstMatrix cdf,
                       TTypes<int32>::ConstVec cdf_size,
                       TTypes<int32>::ConstVec offset,
                       std::string* output) const {
    RangeEncoder encoder;

    DCHECK_GE(cdf.dimension(1), 2);
    DCHECK_LE(cdf.dimension(1), std::numeric_limits<int16>::max());
    DCHECK_EQ(cdf.dimension(0), cdf_size.size());

    const uint32 max_overflow = (1 << overflow_width_) - 1;

    const int64 data_size = data.size();
    for (int64 i = 0; i < data_size; ++i) {
      const int32 cdf_index = index(i);

      DCHECK_GE(cdf_index, 0);
      DCHECK_LT(cdf_index, cdf.dimension(0));

      const int32 max_value = cdf_size(cdf_index) - 2;
      DCHECK_GE(max_value, 0);
      DCHECK_LT(max_value + 1, cdf.dimension(1));

      int32 value = data(i);
      // Map values with tracked probabilities to 0..max_value range.
      value -= offset(cdf_index);
      // If outside of this range, map value to non-negative integer overflow.
      // NOTE: It might be a good idea to check overflow is within uint32 range.
      uint32 overflow = 0;
      if (value < 0) {
        overflow = -2 * value - 1;
        value = max_value;
      } else if (value >= max_value) {
        overflow = 2 * (value - max_value);
        value = max_value;
      }

      const int32* cdf_slice = &cdf(cdf_index, 0);
      encoder.Encode(cdf_slice[value], cdf_slice[value + 1], precision_,
                     output);

      // Encode overflow using variable length code.
      if (value == max_value) {
        int32 widths = 0;
        while (overflow >> (widths * overflow_width_) != 0) {
          ++widths;
        }
        uint32 val = widths;
        while (val >= max_overflow) {
          encoder.Encode(max_overflow, max_overflow + 1, overflow_width_,
                         output);
          val -= max_overflow;
        }
        encoder.Encode(val, val + 1, overflow_width_, output);
        for (int32 j = 0; j < widths; ++j) {
          const uint32 val = (overflow >> (j * overflow_width_)) & max_overflow;
          encoder.Encode(val, val + 1, overflow_width_, output);
        }
      }
    }
    encoder.Finalize(output);
  }

  int precision_;
  int overflow_width_;
  int debug_level_;
};

REGISTER_KERNEL_BUILDER(Name("UnboundedIndexRangeEncode").Device(DEVICE_CPU),
                        UnboundedIndexRangeEncodeOp);

class UnboundedIndexRangeDecodeOp : public OpKernel {
 public:
  explicit UnboundedIndexRangeDecodeOp(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("precision", &precision_));
    OP_REQUIRES(context, 0 < precision_ && precision_ <= 16,
                errors::InvalidArgument("`precision` must be in [1, 16]: ",
                                        precision_));
    OP_REQUIRES_OK(context,
                   context->GetAttr("overflow_width", &overflow_width_));
    OP_REQUIRES(context, 0 < overflow_width_ && overflow_width_ <= 16,
                errors::InvalidArgument("`overflow_width` must be in [1, 16]: ",
                                        overflow_width_));
    OP_REQUIRES_OK(context, context->GetAttr("debug_level", &debug_level_));
    OP_REQUIRES(context, debug_level_ == 0 || debug_level_ == 1,
                errors::InvalidArgument("`debug_level` must be 0 or 1: ",
                                        debug_level_));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& encoded = context->input(0);
    const Tensor& index = context->input(1);
    const Tensor& cdf = context->input(2);
    const Tensor& cdf_size = context->input(3);
    const Tensor& offset = context->input(4);

    OP_REQUIRES(context, encoded.dims() == 0,
                errors::InvalidArgument("`encoded` should be a scalar: ",
                                        encoded.shape()));

    OP_REQUIRES_OK(context, CheckArgumentShapes(index, cdf, cdf_size, offset));
    if (debug_level_ > 0) {
      OP_REQUIRES_OK(context, CheckArgumentValues(precision_, index, cdf,
                                                  cdf_size, offset));
    }

    Tensor* output;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, index.shape(), &output));

    OP_REQUIRES_OK(
        context, RangeDecodeImpl(output->flat<int32>(), index.flat<int32>(),
                                 cdf.matrix<int32>(), cdf_size.vec<int32>(),
                                 offset.vec<int32>(), encoded.flat<tstring>()));
  }

 private:
  tensorflow::Status RangeDecodeImpl(TTypes<int32>::Flat output,
                                     TTypes<int32>::ConstFlat index,
                                     TTypes<int32>::ConstMatrix cdf,
                                     TTypes<int32>::ConstVec cdf_size,
                                     TTypes<int32>::ConstVec offset,
                                     TTypes<tstring>::ConstFlat encoded) const {
    RangeDecoder decoder(encoded(0));

    DCHECK_GE(cdf.dimension(1), 2);
    DCHECK_LE(cdf.dimension(1), std::numeric_limits<int16>::max());

    const uint32 max_overflow = (1 << overflow_width_) - 1;
    const int32 overflow_cdf_size = (1 << overflow_width_) + 1;
    std::vector<int32> overflow_cdf(overflow_cdf_size);
    std::iota(overflow_cdf.begin(), overflow_cdf.end(), 0);

    const int64 output_size = output.size();
    for (int64 i = 0; i < output_size; ++i) {
      const int32 cdf_index = index(i);

      DCHECK_GE(cdf_index, 0);
      DCHECK_LT(cdf_index, cdf.dimension(0));

      const int32 max_value = cdf_size(cdf_index) - 2;
      DCHECK_GE(max_value, 0);
      DCHECK_LT(max_value + 1, cdf.dimension(1));

      const int32* cdf_slice = &cdf(cdf_index, 0);
      int32 value = decoder.Decode(
          absl::Span<const int32>(cdf_slice, max_value + 2), precision_);

      // Decode overflow using variable length code.
      if (value == max_value) {
        int32 widths = 0;
        uint32 val;
        do {
          val = decoder.Decode(overflow_cdf, overflow_width_);
          widths += val;
        } while (val == max_overflow);
        uint32 overflow = 0;
        for (int32 j = 0; j < widths; ++j) {
          const uint32 val = decoder.Decode(overflow_cdf, overflow_width_);
          DCHECK_LE(val, max_overflow);
          overflow |= val << (j * overflow_width_);
        }
        // Map positive values back to integer values.
        value = overflow >> 1;
        if (overflow & 1) {
          value = -value - 1;
        } else {
          value += max_value;
        }
      }

      // Map values in 0..max_range range back to original integer range.
      value += offset(cdf_index);
      output(i) = value;
    }

    return tensorflow::Status::OK();
  }

  int precision_;
  int overflow_width_;
  int debug_level_;
};

REGISTER_KERNEL_BUILDER(Name("UnboundedIndexRangeDecode").Device(DEVICE_CPU),
                        UnboundedIndexRangeDecodeOp);

}  // namespace
}  // namespace tensorflow_compression
