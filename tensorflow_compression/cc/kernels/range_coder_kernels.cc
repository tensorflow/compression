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

#include "tensorflow_compression/cc/kernels/range_coder_kernels.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>

#include "absl/strings/string_view.h"
#include "absl/strings/substitute.h"
#include "absl/types/span.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/tensor_util.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/variant.h"
// variant_encode_decode.h must be included to avoid link error.
#include "tensorflow/core/framework/variant_encode_decode.h"
#include "tensorflow/core/framework/variant_tensor_data.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/statusor.h"
#include "tensorflow/tsl/platform/errors.h"
#include "tensorflow_compression/cc/lib/range_coder.h"

namespace tensorflow_compression {
namespace {

using ::tensorflow::Status;
using ::tensorflow::StatusOr;
using ::tensorflow::Tensor;
using ::tensorflow::TensorShape;
using ::tensorflow::tstring;
using ::tensorflow::TTypes;
using ::tensorflow::Variant;
namespace errors = ::tensorflow::errors;

// -----------------------------------------------------------------------------
// Datatype holder in DT_VARIANT tensors.
// -----------------------------------------------------------------------------
struct EntropyEncoderVariant {
  std::shared_ptr<EntropyEncoderInterface> encoder;

  // These functions are tensorflow::Variant requirements.
  std::string TypeName() const { return "(anonymous)::EntropyEncoderVariant"; }
  void Encode(tensorflow::VariantTensorData* data) const;
  bool Decode(const tensorflow::VariantTensorData& data);
};

void EntropyEncoderVariant::Encode(tensorflow::VariantTensorData* data) const {
  LOG(ERROR) << "Encode() not implemented.";
}

bool EntropyEncoderVariant::Decode(const tensorflow::VariantTensorData& data) {
  LOG(ERROR) << "Decode() not implemented.";
  return false;
}

struct EntropyDecoderVariant {
  std::shared_ptr<EntropyDecoderInterface> decoder;

  // These functions are tensorflow::Variant requirements.
  std::string TypeName() const { return "(anonymous)::EntropyDecoderVariant"; }
  void Encode(tensorflow::VariantTensorData* data) const;
  bool Decode(const tensorflow::VariantTensorData& data);
};

void EntropyDecoderVariant::Encode(tensorflow::VariantTensorData* data) const {
  LOG(ERROR) << "Encode() not implemented.";
}

bool EntropyDecoderVariant::Decode(const tensorflow::VariantTensorData& data) {
  LOG(ERROR) << "Decode() not implemented.";
  return false;
}

// -----------------------------------------------------------------------------
// Utility functions for range coding.
// -----------------------------------------------------------------------------
Status CheckInRange(absl::string_view name, int64_t value, int64_t min,
                    int64_t max) {
  if (value < min || max <= value) {
    return errors::InvalidArgument(
        absl::Substitute("$0=$1 not in range [$2, $3)", name, value, min, max));
  }
  return tensorflow::OkStatus();
}

Status ScanCDF(const int32_t* const end, const int32_t** current,
               std::vector<absl::Span<const int32_t>>* lookup) {
  const int32_t* p = *current;
  if (end < p + 3) {
    // CDF must have at least three values: precision, 0, 1 << precision.
    return errors::InvalidArgument("CDF ended prematurely.");
  }
  const int32_t* precision = p;
  TF_RETURN_IF_ERROR(CheckInRange("precision", std::abs(*precision), 1, 17));
  const int32_t last_value = 1 << std::abs(*precision);
  if (*(++p) != 0) {
    return errors::InvalidArgument("CDF must start with 0.");
  }
  do {
    if (++p == end) {
      return errors::InvalidArgument("CDF must end with 1 << precision.");
    }
    if (p[0] < p[-1]) {
      return errors::InvalidArgument("CDF must be monotonically increasing.");
    }
  } while (*p != last_value);
  lookup->emplace_back(precision, ++p - precision);
  while (p != end && *p == last_value) {
    ++p;
  }
  *current = p;
  return tensorflow::OkStatus();
}

Status IndexCDFVector(const TTypes<int32_t>::ConstFlat& table,
                      std::vector<absl::Span<const int32_t>>* lookup) {
  lookup->clear();
  const int32_t* const start = table.data();
  const int32_t* const end = start + table.size();
  for (const int32_t* current = start; current != end;) {
    TF_RETURN_IF_ERROR(ScanCDF(end, &current, lookup));
  }
  return tensorflow::OkStatus();
}

Status IndexCDFMatrix(const TTypes<int32_t>::ConstMatrix& table,
                      std::vector<absl::Span<const int32_t>>* lookup) {
  lookup->clear();
  lookup->reserve(table.dimension(0));
  const int32_t* const start = table.data();
  const int32_t* const end = start + table.size();
  for (const int32_t* current = start; current != end;) {
    const int32_t* const row_end = current + table.dimension(1);
    TF_RETURN_IF_ERROR(ScanCDF(row_end, &current, lookup));
    if (current != row_end) {
      return errors::InvalidArgument("CDF must end with 1 << precision.");
    }
  }
  return tensorflow::OkStatus();
}

class RangeEncoderInterface : public EntropyEncoderInterface {
 public:
  static tensorflow::StatusOr<std::shared_ptr<RangeEncoderInterface>> Make(
      tensorflow::OpKernelContext* context, const TensorShape& handle_shape) {
    std::shared_ptr<RangeEncoderInterface> p(new RangeEncoderInterface);

    const int64_t n = handle_shape.num_elements();
    p->encoder_.resize(n);
    p->encoded_.resize(n);

    const Tensor& lookup = context->input(1);
    p->lookup_tensor_ = lookup;

    if (lookup.dims() == 1) {
      TF_RETURN_IF_ERROR(IndexCDFVector(lookup.flat<int32_t>(), &p->lookup_));
    } else if (lookup.dims() == 2) {
      TF_RETURN_IF_ERROR(IndexCDFMatrix(lookup.matrix<int32_t>(), &p->lookup_));
    } else {
      TF_RETURN_IF_ERROR(errors::InvalidArgument(
          "`lookup` must be rank 1 or 2: ", lookup.shape()));
    }

    return p;
  }

  Status Encode(tensorflow::OpKernelContext* context,
                TTypes<int32_t>::ConstMatrix index,
                TTypes<int32_t>::ConstMatrix value) override {
    // Sanity checks.
    CHECK_EQ(encoder_.size(), encoded_.size());

    CHECK_EQ(encoder_.size(), value.dimension(0));
    // We may also want to check if lookup_.size() == value_tensor->dim_size(-1)
    // when index tensor was not provided.

    tensorflow::mutex mu;
    tensorflow::Status status ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu);

#define REQUIRE_IN_RANGE(name, value, min, max)     \
  if (auto s = CheckInRange(name, value, min, max); \
      ABSL_PREDICT_FALSE(!s.ok())) {                \
    tensorflow::mutex_lock lock(mu);                \
    status = s;                                     \
    return;                                         \
  }

    const int64_t cost_per_unit = 50 * value.dimension(1);
    tensorflow::thread::ThreadPool* workers =
        context->device()->tensorflow_cpu_worker_threads()->workers;

    if (index.size() != 0) {
      workers->ParallelFor(
          encoder_.size(), cost_per_unit,
          [this, value, index, &mu, &status](int64_t start, int64_t limit) {
            const int64_t lookup_size = lookup_.size();
            const int64_t num_elements = value.dimension(1);
            const int32_t* p_value = &value(start, 0);
            const int32_t* p_index = &index(start, 0);
            for (int64_t i = start; i < limit; ++i) {
              RangeEncoder& encoder = encoder_[i];
              std::string* sink = &encoded_[i];
              for (int64_t j = 0; j < num_elements; ++j) {
                const int32_t ind = *p_index++;
                const int32_t val = *p_value++;

                REQUIRE_IN_RANGE("index", ind, 0, lookup_size);
                absl::Span<const int32_t> row = lookup_[ind];
                // Negative precision value enables overflow functionality.
                if (row[0] > 0) {
                  REQUIRE_IN_RANGE("value", val, 0, row.size() - 2);
                  encoder.Encode(row[val + 1], row[val + 2], row[0], sink);
                } else {
                  OverflowEncode(encoder, sink, row, val);
                }
              }
            }
          });
    } else {
      workers->ParallelFor(
          encoder_.size(), cost_per_unit,
          [this, value, &mu, &status](int64_t start, int64_t limit) {
            const int64_t lookup_size = lookup_.size();
            const int64_t num_elements = value.dimension(1);
            const int32_t* p_value = &value(start, 0);
            for (int64_t i = start; i < limit; ++i) {
              RangeEncoder& encoder = encoder_[i];
              std::string* sink = &encoded_[i];
              for (int64_t ind = 0, j = 0; j < num_elements; ++ind, ++j) {
                const int32_t val = *p_value++;

                ind = (ind < lookup_size) ? ind : 0;
                absl::Span<const int32_t> row = lookup_[ind];
                // Negative precision value enables overflow functionality.
                if (row[0] > 0) {
                  REQUIRE_IN_RANGE("value", val, 0, row.size() - 2);
                  encoder.Encode(row[val + 1], row[val + 2], row[0], sink);
                } else {
                  OverflowEncode(encoder, sink, row, val);
                }
              }
            }
          });
    }
#undef REQUIRE_IN_RANGE

    return status;
  }

  Status Finalize(tensorflow::OpKernelContext* context) override {
    Tensor* tensor;
    TF_RETURN_IF_ERROR(
        context->allocate_output(0, context->input(0).shape(), &tensor));

    auto output = tensor->flat<tstring>();
    CHECK_EQ(encoder_.size(), output.size());

    for (int64_t i = 0; i < output.size(); ++i) {
      encoder_[i].Finalize(&encoded_[i]);
      output(i) = std::move(encoded_[i]);
    }
    return tensorflow::OkStatus();
  }

 private:
  static void OverflowEncode(RangeEncoder& encoder, std::string* sink,
                             const absl::Span<const int32_t> row,
                             int32_t value) {
    const int32_t max_value = row.size() - 3;
    DCHECK_GE(max_value, 0);
    const int32_t sign = value < 0;
    int32_t gamma;
    if (sign) {
      gamma = -value;
      value = max_value;
    } else if (value >= max_value) {
      gamma = value - max_value + 1;
      value = max_value;
    }
    encoder.Encode(row[value + 1], row[value + 2], -row[0], sink);
    // Last interval in CDF table is escape symbol.
    if (value != max_value) {
      return;
    }
    // Encode overflow value using Elias gamma code and binary uniform CDF.
    int32_t n = 1;
    // TODO(ssjhv): Clamp gamma.
    while (gamma >= (1 << n)) {
      encoder.Encode(0, 1, 1, sink);
      ++n;
    }
    while (--n >= 0) {
      const int32_t bit = (gamma >> n) & 1;
      encoder.Encode(bit, bit + 1, 1, sink);
    }
    // Encode sign.
    encoder.Encode(sign, sign + 1, 1, sink);
  }

  RangeEncoderInterface() = default;

  std::vector<absl::Span<const int32_t>> lookup_;
  std::vector<RangeEncoder> encoder_;
  std::vector<std::string> encoded_;
  Tensor lookup_tensor_;  // Ref-count purpose.
};

class RangeDecoderInterface : public EntropyDecoderInterface {
 public:
  static StatusOr<std::shared_ptr<RangeDecoderInterface>> Make(
      tensorflow::OpKernelContext* context) {
    std::shared_ptr<RangeDecoderInterface> p(new RangeDecoderInterface);

    const Tensor& encoded_tensor = context->input(0);
    p->encoded_tensor_ = encoded_tensor;

    auto encoded = encoded_tensor.flat<tstring>();
    for (int64_t i = 0; i < encoded.size(); ++i) {
      p->decoder_.emplace_back(encoded(i));
    }

    const Tensor& lookup = context->input(1);
    p->lookup_tensor_ = lookup;
    if (lookup.dims() == 1) {
      TF_RETURN_IF_ERROR(IndexCDFVector(lookup.flat<int32_t>(), &p->lookup_));
    } else if (lookup.dims() == 2) {
      TF_RETURN_IF_ERROR(IndexCDFMatrix(lookup.matrix<int32_t>(), &p->lookup_));
    } else {
      return errors::InvalidArgument("`lookup` must be rank 1 or 2: ",
                                     lookup.shape());
    }

    return p;
  }

  Status Decode(tensorflow::OpKernelContext* context,
                TTypes<int32_t>::ConstMatrix index,
                TTypes<int32_t>::Matrix output) override {
    // Sanity check.
    CHECK_EQ(decoder_.size(), output.dimension(0));

    tensorflow::mutex mu;
    tensorflow::Status status ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu);

#define REQUIRE_IN_RANGE(name, value, min, max)     \
  if (auto s = CheckInRange(name, value, min, max); \
      ABSL_PREDICT_FALSE(!s.ok())) {                \
    tensorflow::mutex_lock lock(mu);                \
    status = s;                                     \
    return;                                         \
  }

    const int64_t cost_per_unit = 80 * output.dimension(1);
    tensorflow::thread::ThreadPool* workers =
        context->device()->tensorflow_cpu_worker_threads()->workers;

    if (index.size() != 0) {
      workers->ParallelFor(
          decoder_.size(), cost_per_unit,
          [this, index, &output, &mu, &status](int64_t start, int64_t limit) {
            const int64_t lookup_size = lookup_.size();
            const int64_t num_elements = output.dimension(1);
            const int32_t* p_index = &index(start, 0);
            int32_t* p_output = &output(start, 0);
            for (int64_t i = start; i < limit; ++i) {
              RangeDecoder& decoder = decoder_[i];
              for (int64_t j = 0; j < num_elements; ++j) {
                const int32_t ind = *p_index++;
                REQUIRE_IN_RANGE("index", ind, 0, lookup_size);
                absl::Span<const int32_t> row = lookup_[ind];
                // Negative precision value enables overflow functionality.
                if (row[0] > 0) {
                  *p_output++ = decoder.Decode(row.subspan(1), row[0]);
                } else {
                  *p_output++ = OverflowDecode(decoder, row);
                }
              }
            }
          });
    } else {
      workers->ParallelFor(
          decoder_.size(), cost_per_unit,
          [this, &output](int64_t start, int64_t limit) {
            const int64_t lookup_size = lookup_.size();
            const int64_t num_elements = output.dimension(1);
            int32_t* p_output = &output(start, 0);
            for (int64_t i = start; i < limit; ++i) {
              RangeDecoder& decoder = decoder_[i];
              for (int64_t ind = 0, j = 0; j < num_elements; ++ind, ++j) {
                if (lookup_size <= ind) ind = 0;
                absl::Span<const int32_t> row = lookup_[ind];
                // Negative precision value enables overflow functionality.
                if (row[0] > 0) {
                  *p_output++ = decoder.Decode(row.subspan(1), row[0]);
                } else {
                  *p_output++ = OverflowDecode(decoder, row);
                }
              }
            }
          });
    }

#undef REQUIRE_IN_RANGE
    return status;
  }

  Status Finalize(tensorflow::OpKernelContext* context) override {
    Tensor* output_tensor;
    TF_RETURN_IF_ERROR(
        context->allocate_output(0, encoded_tensor_.shape(), &output_tensor));
    auto output = output_tensor->flat<bool>();
    output.setConstant(true);

    CHECK_EQ(output.size(), decoder_.size());
    for (int64_t i = 0; i < output.size(); ++i) {
      if (bool success = decoder_[i].Finalize(); !success) {
        output(i) = false;
        VLOG(0) << "RangeDecoder #" << i << " final status was an error";
      }
    }
    return tensorflow::OkStatus();
  }

 private:
  static int32_t OverflowDecode(RangeDecoder& decoder,
                                const absl::Span<const int32_t> row) {
    constexpr int32_t binary_uniform_cdf[] = {0, 1, 2};
    const int32_t max_value = row.size() - 3;
    DCHECK_GE(max_value, 0);
    int32_t value = decoder.Decode(row.subspan(1), -row[0]);
    // Last interval in CDF table is escape symbol.
    if (value != max_value) {
      return value;
    }
    // Decode overflow using Elias gamma code and binary uniform CDF.
    int32_t n = 0;
    while (decoder.DecodeLinearly(binary_uniform_cdf, 1) == 0) {
      ++n;
    }
    value = 1 << n;
    while (--n >= 0) {
      value |= decoder.DecodeLinearly(binary_uniform_cdf, 1) << n;
    }
    // Decode sign.
    const int32_t sign = decoder.DecodeLinearly(binary_uniform_cdf, 1);
    return sign ? -value : value + max_value - 1;
  }

  RangeDecoderInterface() = default;

  std::vector<absl::Span<const int32_t>> lookup_;
  std::vector<RangeDecoder> decoder_;
  Tensor encoded_tensor_;  // Ref-count purpose.
  Tensor lookup_tensor_;   // Ref-count purpose.
};

// -----------------------------------------------------------------------------
// RangeEncoder ops
// -----------------------------------------------------------------------------
template <typename Interface>
class CreateRangeEncoderOp : public tensorflow::OpKernel {
 public:
  using OpKernel::OpKernel;

  void Compute(tensorflow::OpKernelContext* context) override {
    TensorShape handle_shape;
    OP_REQUIRES_OK(context, tensorflow::tensor::MakeShape(context->input(0),
                                                          &handle_shape));

    EntropyEncoderVariant wrap;
    OP_REQUIRES_VALUE(wrap.encoder, context,
                      Interface::Make(context, handle_shape));

    Tensor* output_tensor;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, handle_shape, &output_tensor));
    output_tensor->flat<Variant>()(0) = std::move(wrap);
  }
};

REGISTER_KERNEL_BUILDER(
    Name("CreateRangeEncoder").Device(tensorflow::DEVICE_CPU),
    CreateRangeEncoderOp<RangeEncoderInterface>);

class EntropyEncodeOp : public tensorflow::OpKernel {
 public:
  using OpKernel::OpKernel;

  void Compute(tensorflow::OpKernelContext* context) override {
    // This is an unnecessary shallow copy but helps avoiding a const_cast.
    Tensor handle = context->input(0);
    const TensorShape& handle_shape = handle.shape();
    OP_REQUIRES(context, handle_shape.num_elements() != 0,
                errors::InvalidArgument("`handle` is empty: handle.shape=",
                                        handle_shape));

    auto* wrap = handle.flat<Variant>()(0).get<EntropyEncoderVariant>();
    OP_REQUIRES(context, wrap != nullptr && wrap->encoder != nullptr,
                errors::InvalidArgument("'handle' is not an encoder"));

    Tensor const* value_tensor = nullptr;
    OP_REQUIRES_OK(context, context->input("value", &value_tensor));

    OP_REQUIRES(
        context,
        tensorflow::TensorShapeUtils::StartsWith(value_tensor->shape(),
                                                 handle_shape),
        errors::InvalidArgument(
            "'value' shape should start with 'handle' shape: value.shape=",
            value_tensor->shape(),
            " does not start with handle.shape=", handle_shape));

    const int prefix_dims = handle_shape.dims();
    auto value =
        value_tensor->flat_inner_outer_dims<int32_t, 2>(prefix_dims - 1);

    if (context->num_inputs() > 2) {
      // EntropyEncodeIndex op.
      Tensor const* index_tensor = nullptr;
      OP_REQUIRES_OK(context, context->input("index", &index_tensor));
      OP_REQUIRES(context, index_tensor->shape() == value_tensor->shape(),
                  errors::InvalidArgument(
                      "'index' shape should match 'value' shape: index.shape=",
                      index_tensor->shape(),
                      " != value.shape=", value_tensor->shape()));

      auto index =
          index_tensor->flat_inner_outer_dims<int32_t, 2>(prefix_dims - 1);
      OP_REQUIRES_OK(context, wrap->encoder->Encode(context, index, value));
    } else {
      // EntropyEncodeChannel op.
      TTypes<int32_t>::ConstMatrix index(nullptr, 0, 0);
      OP_REQUIRES_OK(context, wrap->encoder->Encode(context, index, value));
    }

    context->set_output(0, handle);
  }
};

REGISTER_KERNEL_BUILDER(
    Name("EntropyEncodeIndex").Device(tensorflow::DEVICE_CPU), EntropyEncodeOp);
REGISTER_KERNEL_BUILDER(
    Name("EntropyEncodeChannel").Device(tensorflow::DEVICE_CPU),
    EntropyEncodeOp);

class EntropyEncodeFinalizeOp : public tensorflow::OpKernel {
 public:
  using OpKernel::OpKernel;

  void Compute(tensorflow::OpKernelContext* context) override {
    // This is an unnecessary shallow copy but helps avoiding a const_cast.
    Tensor handle_tensor = context->input(0);
    OP_REQUIRES(
        context, handle_tensor.shape().num_elements() != 0,
        errors::InvalidArgument("`handle` is empty: ", handle_tensor.shape()));

    auto* wrap = handle_tensor.flat<Variant>()(0).get<EntropyEncoderVariant>();
    OP_REQUIRES(context, wrap != nullptr && wrap->encoder != nullptr,
                errors::InvalidArgument("'handle' is not an encoder"));

    // Finalize() should take care of the output.
    OP_REQUIRES_OK(context, wrap->encoder->Finalize(context));
  }
};

REGISTER_KERNEL_BUILDER(
    Name("EntropyEncodeFinalize").Device(tensorflow::DEVICE_CPU),
    EntropyEncodeFinalizeOp);

// -----------------------------------------------------------------------------
// RangeDecoder ops
// -----------------------------------------------------------------------------
template <typename Interface>
class CreateRangeDecoderOp : public tensorflow::OpKernel {
 public:
  using OpKernel::OpKernel;

  void Compute(tensorflow::OpKernelContext* context) override {
    TensorShape handle_shape = context->input(0).shape();
    OP_REQUIRES(context, handle_shape.num_elements() != 0,
                errors::InvalidArgument("`encoded` is empty: ", handle_shape));

    EntropyDecoderVariant wrap;
    OP_REQUIRES_VALUE(wrap.decoder, context,
                      Interface::Make(context));

    Tensor* output;
    OP_REQUIRES_OK(context, context->allocate_output(0, handle_shape, &output));
    output->flat<Variant>()(0) = std::move(wrap);
  }
};

REGISTER_KERNEL_BUILDER(
    Name("CreateRangeDecoder").Device(tensorflow::DEVICE_CPU),
    CreateRangeDecoderOp<RangeDecoderInterface>);

class EntropyDecodeOp : public tensorflow::OpKernel {
 public:
  using OpKernel::OpKernel;

  void Compute(tensorflow::OpKernelContext* context) override {
    // This is an unnecessary shallow copy but helps avoiding a const_cast.
    Tensor handle = context->input(0);
    OP_REQUIRES(context, handle.shape().num_elements() != 0,
                errors::InvalidArgument("`handle` is empty: ", handle.shape()));
    context->set_output(0, handle);

    auto* wrap = handle.flat<Variant>()(0).get<EntropyDecoderVariant>();
    OP_REQUIRES(context, wrap != nullptr && wrap->decoder != nullptr,
                errors::InvalidArgument("'handle' is not a decoder"));

    Tensor const* suffix_shape_tensor = nullptr;
    OP_REQUIRES_OK(context, context->input("shape", &suffix_shape_tensor));

    TensorShape suffix_shape;
    OP_REQUIRES_OK(context, tensorflow::tensor::MakeShape(*suffix_shape_tensor,
                                                          &suffix_shape));

    TensorShape output_shape = handle.shape();
    output_shape.AppendShape(suffix_shape);

    Tensor* output_tensor;
    OP_REQUIRES_OK(context,
                   context->allocate_output(1, output_shape, &output_tensor));
    auto output =
        output_tensor->flat_inner_outer_dims<int32_t, 2>(handle.dims() - 1);

    if (context->num_inputs() > 2) {
      // EntropyDecodeIndex op.
      Tensor const* index_tensor = nullptr;
      OP_REQUIRES_OK(context, context->input("index", &index_tensor));
      OP_REQUIRES(context, index_tensor->shape() == output_shape,
                  errors::InvalidArgument("'index' shape should match 'handle' "
                                          "shape + 'shape': index.shape=",
                                          index_tensor->shape(),
                                          ", handle.shape=", handle.shape(),
                                          ", shape=", suffix_shape));

      auto index =
          index_tensor->flat_inner_outer_dims<int32_t, 2>(handle.dims() - 1);
      OP_REQUIRES_OK(context, wrap->decoder->Decode(context, index, output));
    } else {
      // EntropyDecodeChannel op.
      TTypes<int32_t>::ConstMatrix index(nullptr, 0, 0);
      OP_REQUIRES_OK(context, wrap->decoder->Decode(context, index, output));
    }
  }
};

REGISTER_KERNEL_BUILDER(
    Name("EntropyDecodeIndex").Device(tensorflow::DEVICE_CPU), EntropyDecodeOp);
REGISTER_KERNEL_BUILDER(
    Name("EntropyDecodeChannel").Device(tensorflow::DEVICE_CPU),
    EntropyDecodeOp);

class EntropyDecodeFinalizeOp : public tensorflow::OpKernel {
 public:
  using OpKernel::OpKernel;

  void Compute(tensorflow::OpKernelContext* context) override {
    // This is an unnecessary shallow copy but helps avoiding a const_cast.
    Tensor handle = context->input(0);
    OP_REQUIRES(context, handle.shape().num_elements() != 0,
                errors::InvalidArgument("`handle` is empty: ", handle.shape()));

    auto* wrap = handle.flat<Variant>()(0).get<EntropyDecoderVariant>();
    OP_REQUIRES(context, wrap != nullptr && wrap->decoder != nullptr,
                errors::InvalidArgument("'handle' is not a decoder"));

    OP_REQUIRES_OK(context, wrap->decoder->Finalize(context));
  }
};

REGISTER_KERNEL_BUILDER(
    Name("EntropyDecodeFinalize").Device(tensorflow::DEVICE_CPU),
    EntropyDecodeFinalizeOp);

}  // namespace
}  // namespace tensorflow_compression
