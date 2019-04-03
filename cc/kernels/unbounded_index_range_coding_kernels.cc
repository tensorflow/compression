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

#define EIGEN_USE_THREADS

#include <algorithm>
#include <array>
#include <limits>
#include <type_traits>
#include <vector>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/types.h"

#include "tensorflow_compression/cc/kernels/range_coder.h"

namespace tensorflow_compression {
namespace {
namespace errors = tensorflow::errors;
namespace gtl = tensorflow::gtl;
using tensorflow::DEVICE_CPU;
using tensorflow::OpKernel;
using tensorflow::OpKernelConstruction;
using tensorflow::OpKernelContext;
using tensorflow::Status;
using tensorflow::Tensor;
using tensorflow::TensorShape;
using tensorflow::TensorShapeUtils;
using tensorflow::TTypes;

// Non-incremental encoder op -------------------------------------------------
class UnboundedIndexRangeEncodeOp : public OpKernel {
 public:
  explicit UnboundedIndexRangeEncodeOp(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("precision", &precision_));
    OP_REQUIRES_OK(context,
                   context->GetAttr("overflow_width", &overflow_width_));
    OP_REQUIRES(context, 0 < precision_ && precision_ <= 16,
                errors::InvalidArgument("`precision` must be in [1, 16]: ",
                                        precision_));
    OP_REQUIRES(
        context, 0 < overflow_width_ && overflow_width_ <= precision_,
        errors::InvalidArgument("`overflow_width` must be in [1, precision]: ",
                                overflow_width_));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& data = context->input(0);
    const Tensor& index = context->input(1);
    const Tensor& cdf = context->input(2);
    const Tensor& cdf_size = context->input(3);
    const Tensor& offset = context->input(4);

    OP_REQUIRES(context, data.shape() == index.shape(),
                errors::InvalidArgument(
                    "`data` and `index` should have the same shape"));

    OP_REQUIRES(context, TensorShapeUtils::IsMatrix(cdf.shape()),
                errors::InvalidArgument("`cdf` should be 2-D."));
    OP_REQUIRES(
        context,
        TensorShapeUtils::IsVector(cdf_size.shape()) &&
            cdf_size.dim_size(0) == cdf.dim_size(0),
        errors::InvalidArgument("`cdf_size` should be 1-D and its length "
                                "should match the number of rows in `cdf`."));
    OP_REQUIRES(
        context,
        TensorShapeUtils::IsVector(offset.shape()) &&
            offset.dim_size(0) == cdf.dim_size(0),
        errors::InvalidArgument("`offset` should be 1-D and its length "
                                "should match the number of rows in `cdf`."));

    Tensor* output;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, TensorShape{}, &output));

    RangeEncodeImpl(data.flat<int32>(), index.flat<int32>(),
                    cdf.matrix<int32>(), cdf_size.vec<int32>(),
                    offset.vec<int32>(), &output->flat<string>()(0));
  }

 private:
  void RangeEncodeImpl(TTypes<int32>::ConstFlat data,
                       TTypes<int32>::ConstFlat index,
                       TTypes<int32>::ConstMatrix cdf,
                       TTypes<int32>::ConstVec cdf_size,
                       TTypes<int32>::ConstVec offset, string* output) const {
    RangeEncoder encoder{precision_};

    DCHECK_GE(cdf.dimension(1), 2);
    DCHECK_LE(cdf.dimension(1), std::numeric_limits<int16>::max());
    DCHECK_EQ(cdf.dimension(0), cdf_size.size());

    const uint32 max_overflow = (1 << overflow_width_) - 1;
    const uint32 overflow_shift = precision_ - overflow_width_;

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
      uint32 overflow;
      if (value < 0) {
        overflow = -2 * value - 1;
        value = max_value;
      } else if (value >= max_value) {
        overflow = 2 * (value - max_value);
        value = max_value;
      }

      const int32* cdf_slice = &cdf(cdf_index, 0);
      encoder.Encode(cdf_slice[value], cdf_slice[value + 1], output);

      // Encode overflow using variable length code.
      if (value == max_value) {
        int32 widths = 0;
        while (overflow >> (widths * overflow_width_)) {
          ++widths;
        }
        uint32 val = widths;
        while (val >= max_overflow) {
          encoder.Encode(max_overflow << overflow_shift,
                         (max_overflow + 1) << overflow_shift, output);
          val -= max_overflow;
        }
        encoder.Encode(val << overflow_shift, (val + 1) << overflow_shift,
                       output);
        for (int32 j = 0; j < widths; ++j) {
          const uint32 val = (overflow >> (j * overflow_width_)) & max_overflow;
          encoder.Encode(val << overflow_shift, (val + 1) << overflow_shift,
                         output);
        }
      }
    }
    encoder.Finalize(output);
  }

  int precision_;
  int overflow_width_;
};

REGISTER_KERNEL_BUILDER(Name("UnboundedIndexRangeEncode").Device(DEVICE_CPU),
                        UnboundedIndexRangeEncodeOp);

// Non-incremental decoder op -------------------------------------------------
class UnboundedIndexRangeDecodeOp : public OpKernel {
 public:
  explicit UnboundedIndexRangeDecodeOp(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("precision", &precision_));
    OP_REQUIRES_OK(context,
                   context->GetAttr("overflow_width", &overflow_width_));
    OP_REQUIRES(context, 0 < precision_ && precision_ <= 16,
                errors::InvalidArgument("`precision` must be in [1, 16]: ",
                                        precision_));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& encoded = context->input(0);
    const Tensor& index = context->input(1);
    const Tensor& cdf = context->input(2);
    const Tensor& cdf_size = context->input(3);
    const Tensor& offset = context->input(4);

    OP_REQUIRES(context, encoded.shape() == TensorShape{},
                errors::InvalidArgument("Invalid `encoded` shape: ",
                                        encoded.shape().DebugString()));
    OP_REQUIRES(context, TensorShapeUtils::IsMatrix(cdf.shape()),
                errors::InvalidArgument("`cdf` should be 2-D."));
    OP_REQUIRES(
        context,
        TensorShapeUtils::IsVector(cdf_size.shape()) &&
            cdf_size.dim_size(0) == cdf.dim_size(0),
        errors::InvalidArgument("`cdf_size` should be 1-D and its length "
                                "should match the number of rows in `cdf`."));
    OP_REQUIRES(
        context,
        TensorShapeUtils::IsVector(offset.shape()) &&
            offset.dim_size(0) == cdf.dim_size(0),
        errors::InvalidArgument("`offset` should be 1-D and its length "
                                "should match the number of rows in `cdf`."));

    Tensor* output;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, index.shape(), &output));

    OP_REQUIRES_OK(
        context, RangeDecodeImpl(output->flat<int32>(), index.flat<int32>(),
                                 cdf.matrix<int32>(), cdf_size.vec<int32>(),
                                 offset.vec<int32>(), encoded.flat<string>()));
  }

 private:
  tensorflow::Status RangeDecodeImpl(TTypes<int32>::Flat output,
                                     TTypes<int32>::ConstFlat index,
                                     TTypes<int32>::ConstMatrix cdf,
                                     TTypes<int32>::ConstVec cdf_size,
                                     TTypes<int32>::ConstVec offset,
                                     TTypes<string>::ConstFlat encoded) const {
    RangeDecoder decoder{encoded(0), precision_};

    DCHECK_GE(cdf.dimension(1), 2);
    DCHECK_LE(cdf.dimension(1), std::numeric_limits<int16>::max());

    const uint32 max_overflow = (1 << overflow_width_) - 1;
    const int32 overflow_cdf_size = (1 << overflow_width_) + 1;
    std::vector<int32> overflow_cdf(overflow_cdf_size);
    for (int32 i = 0; i < overflow_cdf_size; ++i) {
      overflow_cdf[i] = i << (precision_ - overflow_width_);
    }

    const int64 output_size = output.size();
    for (int64 i = 0; i < output_size; ++i) {
      const int32 cdf_index = index(i);

      DCHECK_GE(cdf_index, 0);
      DCHECK_LT(cdf_index, cdf.dimension(0));

      const int32 max_value = cdf_size(cdf_index) - 2;
      DCHECK_GE(max_value, 0);
      DCHECK_LT(max_value + 1, cdf.dimension(1));

      const int32* cdf_slice = &cdf(cdf_index, 0);
      int32 value =
          decoder.Decode(gtl::ArraySlice<int32>(cdf_slice, max_value + 2));

      // Decode overflow using variable length code.
      if (value == max_value) {
        int32 widths = 0;
        uint32 val;
        do {
          val = decoder.Decode(overflow_cdf);
          widths += val;
        } while (val == max_overflow);
        uint32 overflow = 0;
        for (int32 j = 0; j < widths; ++j) {
          const uint32 val = decoder.Decode(overflow_cdf);
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
};

REGISTER_KERNEL_BUILDER(Name("UnboundedIndexRangeDecode").Device(DEVICE_CPU),
                        UnboundedIndexRangeDecodeOp);

}  // namespace
}  // namespace tensorflow_compression
