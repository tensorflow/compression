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

#include <cstdint>
#include <string>
#include <utility>

#include "absl/base/integral_types.h"
#include "tensorflow_compression/cc/lib/range_coder.h"
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

namespace tensorflow_compression {
namespace {

using ::tensorflow::Status;
using ::tensorflow::Tensor;
using ::tensorflow::TensorShape;
using ::tensorflow::tstring;
using ::tensorflow::TTypes;
using ::tensorflow::Variant;
namespace errors = ::tensorflow::errors;

Status CheckInRange(absl::string_view name, int64_t value, int64_t min,
                    int64_t max) {
  if (value < min || max <= value) {
    return errors::InvalidArgument(
        absl::Substitute("$0=$1 not in range [$2, $3)", name, value, min, max));
  }
  return Status::OK();
}

Status ScanCDF(const int32_t* const end, const int32_t** current,
               std::vector<absl::Span<const int32_t>>* lookup) {
  const int32_t* p = *current;
  if (end < p + 3) {
    // CDF must have at least values: precision, 0, 1 << precision.
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
  return Status::OK();
}

Status IndexCDFVector(const TTypes<int32_t>::ConstFlat& table,
                      std::vector<absl::Span<const int32_t>>* lookup) {
  lookup->clear();
  const int32_t* const start = table.data();
  const int32_t* const end = start + table.size();
  for (const int32_t* current = start; current != end;) {
    TF_RETURN_IF_ERROR(ScanCDF(end, &current, lookup));
  }
  return Status::OK();
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
  return Status::OK();
}

class RangeEncoderInterface final : public EntropyEncoderInterface {
 public:
  static Status MakeShared(const Tensor lookup,
                           std::shared_ptr<EntropyEncoderInterface>* ptr) {
    Status status;
    RangeEncoderInterface* re = new RangeEncoderInterface(lookup);
    if (lookup.dims() == 1) {
      status = IndexCDFVector(lookup.flat<int32_t>(), &re->lookup_);
    } else if (lookup.dims() == 2) {
      status = IndexCDFMatrix(lookup.matrix<int32_t>(), &re->lookup_);
    } else {
      status = errors::InvalidArgument("`lookup` must be rank 1 or 2.");
    }
    if (status.ok()) {
      ptr->reset(re);
    } else {
      delete re;
    }
    return status;
  }

  Status Encode(int32_t index, int32_t value) override {
    TF_RETURN_IF_ERROR(CheckInRange("index", index, 0, lookup_.size()));
    absl::Span<const int32_t> row = lookup_[index];
    // Negative precision value enables overflow functionality.
    if (row[0] > 0) {
      TF_RETURN_IF_ERROR(CheckInRange("value", value, 0, row.size() - 2));
      encoder_.Encode(row[value + 1], row[value + 2], row[0], &encoded_);
    } else {
      OverflowEncode(row, value);
    }
    return Status::OK();
  }

  Status Finalize(std::string* sink) override {
    encoder_.Finalize(&encoded_);
    *sink = std::move(encoded_);
    return Status::OK();
  }

 private:
  explicit RangeEncoderInterface(Tensor lookup) : hold_(std::move(lookup)) {}

  void OverflowEncode(const absl::Span<const int32_t> row, int32_t value) {
    const int32_t max_value = row.size() - 3;
    const int32_t sign = value < 0;
    int32_t gamma;
    if (sign) {
      gamma = -value;
      value = max_value;
    } else if (value >= max_value) {
      gamma = value - max_value + 1;
      value = max_value;
    }
    encoder_.Encode(row[value + 1], row[value + 2], -row[0], &encoded_);
    // Last interval in CDF table is escape symbol.
    if (value != max_value) {
      return;
    }
    // Encode overflow value using Elias gamma code and binary uniform CDF.
    int32_t n = 1;
    while (gamma >= (1 << n)) {
      encoder_.Encode(0, 1, 1, &encoded_);
      ++n;
    }
    while (--n >= 0) {
      const int32_t bit = (gamma >> n) & 1;
      encoder_.Encode(bit, bit + 1, 1, &encoded_);
    }
    // Encode sign.
    encoder_.Encode(sign, sign + 1, 1, &encoded_);
  }

  std::vector<absl::Span<const int32_t>> lookup_;
  RangeEncoder encoder_;
  std::string encoded_;
  Tensor hold_;
};

class RangeDecoderInterface final : public EntropyDecoderInterface {
 public:
  static Status MakeShared(absl::string_view encoded, const Tensor lookup,
                           std::shared_ptr<EntropyDecoderInterface>* ptr) {
    Status status;
    RangeDecoderInterface* rd = new RangeDecoderInterface(encoded, lookup);
    if (lookup.dims() == 1) {
      status = IndexCDFVector(lookup.flat<int32_t>(), &rd->lookup_);
    } else if (lookup.dims() == 2) {
      status = IndexCDFMatrix(lookup.matrix<int32_t>(), &rd->lookup_);
    } else {
      status = errors::InvalidArgument("`lookup` must be rank 1 or 2.");
    }
    if (status.ok()) {
      ptr->reset(rd);
    } else {
      delete rd;
    }
    return status;
  }

  Status Decode(int32_t index, int32_t* output) override {
    TF_RETURN_IF_ERROR(CheckInRange("index", index, 0, lookup_.size()));
    absl::Span<const int32_t> row = lookup_[index];
    // Negative precision value enables overflow functionality.
    if (row[0] > 0) {
      *output = decoder_.Decode(row.subspan(1), row[0]);
    } else {
      *output = OverflowDecode(row);
    }
    return Status::OK();
  }

  Status Finalize() override {
    if (!decoder_.Finalize()) {
      return errors::DataLoss("RangeDecoder returned an error status");
    }
    return Status::OK();
  }

 private:
  RangeDecoderInterface(absl::string_view encoded, Tensor lookup)
      : decoder_(encoded), hold_(std::move(lookup)) {}

  int32_t OverflowDecode(const absl::Span<const int32_t> row) {
    constexpr int32_t binary_uniform_cdf[] = {0, 1, 2};
    const int32_t max_value = row.size() - 3;
    int32_t value = decoder_.Decode(row.subspan(1), -row[0]);
    // Last interval in CDF table is escape symbol.
    if (value != max_value) {
      return value;
    }
    // Decode overflow using Elias gamma code and binary uniform CDF.
    int32_t n = 0;
    while (decoder_.DecodeLinearly(binary_uniform_cdf, 1) == 0) {
      ++n;
    }
    value = 1 << n;
    while (--n >= 0) {
      value |= decoder_.DecodeLinearly(binary_uniform_cdf, 1) << n;
    }
    // Decode sign.
    const int32_t sign = decoder_.DecodeLinearly(binary_uniform_cdf, 1);
    return sign ? -value : value + max_value - 1;
  }

  std::vector<absl::Span<const int32_t>> lookup_;
  RangeDecoder decoder_;
  Tensor hold_;
};

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
  Tensor holder;

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

// RangeEncoder ops -----------------------------------------------------------
class CreateRangeEncoderOp : public tensorflow::OpKernel {
 public:
  using OpKernel::OpKernel;
  void Compute(tensorflow::OpKernelContext* context) override {
    TensorShape handle_shape;
    OP_REQUIRES_OK(context, tensorflow::tensor::MakeShape(context->input(0),
                                                          &handle_shape));

    Tensor* output_tensor;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, handle_shape, &output_tensor));

    const Tensor& lookup = context->input(1);
    auto output = output_tensor->flat<Variant>();
    for (int64_t i = 0; i < output.size(); ++i) {
      EntropyEncoderVariant wrap;
      OP_REQUIRES_OK(context,
                     RangeEncoderInterface::MakeShared(lookup, &wrap.encoder));
      output(i) = std::move(wrap);
    }
  }
};

REGISTER_KERNEL_BUILDER(
    Name("CreateRangeEncoder").Device(tensorflow::DEVICE_CPU),
    CreateRangeEncoderOp);

class EntropyEncodeChannelOp : public tensorflow::OpKernel {
 public:
  using OpKernel::OpKernel;

  tensorflow::Status CheckShapes(tensorflow::OpKernelContext* context) const {
    const TensorShape& handle_shape = context->input(0).shape();
    const TensorShape& value_shape = context->input(1).shape();
    if (!tensorflow::TensorShapeUtils::StartsWith(value_shape, handle_shape)) {
      return errors::InvalidArgument(
          "'value' shape should start with 'handle' shape: value.shape=",
          value_shape, " does not start with handle.shape=", handle_shape);
    }
    return tensorflow::Status::OK();
  }

  void Compute(tensorflow::OpKernelContext* context) override {
    OP_REQUIRES_OK(context, CheckShapes(context));

    // This is an unnecessary shallow copy but helps avoiding a const_cast.
    Tensor handle_tensor = context->input(0);
    auto handle = handle_tensor.flat<Variant>();

    const int prefix_dims = handle_tensor.dims();
    auto value =
        context->input(1).flat_inner_outer_dims<int32_t, 2>(prefix_dims - 1);
    CHECK_EQ(handle.dimension(0), value.dimension(0));

    const TensorShape& value_shape = context->input(1).shape();
    const int64_t index_stride =
        value_shape.dims() == prefix_dims
            ? 1
            : value_shape.dim_size(value_shape.dims() - 1);
    CHECK_EQ(value.dimension(1) % index_stride, 0);

    const int64_t cost_per_unit = 50 * value.dimension(1);

    tensorflow::thread::ThreadPool* workers =
        context->device()->tensorflow_cpu_worker_threads()->workers;
    tensorflow::mutex mu;
    workers->ParallelFor(
        handle.size(), cost_per_unit,
        [&handle, &mu, context, value, index_stride](int64 start, int64 limit) {
          PerShard(handle, value, index_stride, context, &mu, start, limit);
        });

    context->set_output(0, handle_tensor);
  }

 private:
  static void PerShard(TTypes<Variant>::Flat handle,
                       TTypes<int32_t>::ConstMatrix value,
                       const int64_t index_stride,
                       tensorflow::OpKernelContext* context,
                       tensorflow::mutex* mu, int64_t start, int64_t limit) {
#define REQUIRES_OK(status)                             \
  if (auto s = (status); ABSL_PREDICT_FALSE(!s.ok())) { \
    tensorflow::mutex_lock lock(*mu);                   \
    context->SetStatus(s);                              \
    return;                                             \
  }

#define REQUIRES(cond, status)        \
  if (!ABSL_PREDICT_TRUE(cond)) {     \
    tensorflow::mutex_lock lock(*mu); \
    context->SetStatus(status);       \
    return;                           \
  }

    const int64_t num_elements = value.dimension(1);
    auto* p_value = &value(start, 0);
    int64_t index = 0;

    for (int64_t i = start; i < limit; ++i) {
      auto* wrap = handle(i).get<EntropyEncoderVariant>();
      REQUIRES(wrap != nullptr && wrap->encoder != nullptr,
               errors::InvalidArgument("'handle' is not an encoder"));
      auto* encoder = wrap->encoder.get();

      for (int64_t j = 0; j < num_elements; ++j) {
        REQUIRES_OK(encoder->Encode(index++, *(p_value++)));
        if (index == index_stride) index = 0;
      }
    }

#undef REQUIRES
#undef REQUIRES_OK
  }
};

REGISTER_KERNEL_BUILDER(
    Name("EntropyEncodeChannel").Device(tensorflow::DEVICE_CPU),
    EntropyEncodeChannelOp);

class EntropyEncodeIndexOp : public tensorflow::OpKernel {
 public:
  using OpKernel::OpKernel;

  tensorflow::Status CheckShapes(tensorflow::OpKernelContext* context) const {
    const TensorShape& handle_shape = context->input(0).shape();
    const TensorShape& index_shape = context->input(1).shape();
    const TensorShape& value_shape = context->input(2).shape();

    if (value_shape != index_shape) {
      return errors::InvalidArgument(
          "'value' shape should match 'index' shape: value.shape=", value_shape,
          " != index.shape=", index_shape);
    }

    if (!tensorflow::TensorShapeUtils::StartsWith(index_shape, handle_shape)) {
      return errors::InvalidArgument(
          "'index' shape should start with 'handle' shape: index.shape=",
          index_shape, " does not start with handle.shape=", handle_shape);
    }

    return tensorflow::Status::OK();
  }

  void Compute(tensorflow::OpKernelContext* context) override {
    OP_REQUIRES_OK(context, CheckShapes(context));

    // This is an unnecessary shallow copy but helps avoiding a const_cast.
    Tensor handle_tensor = context->input(0);
    auto handle = handle_tensor.flat<Variant>();

    const int prefix_dims = handle_tensor.dims();
    auto index =
        context->input(1).flat_inner_outer_dims<int32_t, 2>(prefix_dims - 1);
    auto value =
        context->input(2).flat_inner_outer_dims<int32_t, 2>(prefix_dims - 1);

    CHECK_EQ(handle.dimension(0), value.dimension(0));

    const int64_t cost_per_unit = 50 * value.dimension(1);

    tensorflow::thread::ThreadPool* workers =
        context->device()->tensorflow_cpu_worker_threads()->workers;
    tensorflow::mutex mu;
    workers->ParallelFor(
        handle.size(), cost_per_unit,
        [&handle, &mu, context, value, index](int64 start, int64 limit) {
          PerShard(handle, index, value, context, &mu, start, limit);
        });

    context->set_output(0, handle_tensor);
  }

 private:
  static void PerShard(TTypes<Variant>::Flat handle,
                       TTypes<int32_t>::ConstMatrix index,
                       TTypes<int32_t>::ConstMatrix value,
                       tensorflow::OpKernelContext* context,
                       tensorflow::mutex* mu, int64_t start, int64_t limit) {
#define REQUIRES_OK(status)                             \
  if (auto s = (status); ABSL_PREDICT_FALSE(!s.ok())) { \
    tensorflow::mutex_lock lock(*mu);                   \
    context->SetStatus(s);                              \
    return;                                             \
  }

#define REQUIRES(cond, status)        \
  if (!ABSL_PREDICT_TRUE(cond)) {     \
    tensorflow::mutex_lock lock(*mu); \
    context->SetStatus(status);       \
    return;                           \
  }

    const int64_t num_elements = value.dimension(1);
    const int32_t* p_value = &value(start, 0);
    const int32_t* p_index = &index(start, 0);

    for (int64_t i = start; i < limit; ++i) {
      auto* wrap = handle(i).get<EntropyEncoderVariant>();
      REQUIRES(wrap != nullptr && wrap->encoder != nullptr,
               errors::InvalidArgument("'handle' is not an encoder"));
      auto* encoder = wrap->encoder.get();

      for (int64_t j = 0; j < num_elements; ++j) {
        REQUIRES_OK(encoder->Encode(*(p_index++), *(p_value++)));
      }
    }

#undef REQUIRES
#undef REQUIRES_OK
  }
};

REGISTER_KERNEL_BUILDER(
    Name("EntropyEncodeIndex").Device(tensorflow::DEVICE_CPU),
    EntropyEncodeIndexOp);

class EntropyEncodeFinalizeOp : public tensorflow::OpKernel {
 public:
  using OpKernel::OpKernel;

  void Compute(tensorflow::OpKernelContext* context) override {
    // This is an unnecessary shallow copy but helps avoiding a const_cast.
    Tensor handle_tensor = context->input(0);
    auto handle = handle_tensor.flat<Variant>();

    Tensor* output_tensor;
    OP_REQUIRES_OK(context, context->allocate_output(0, handle_tensor.shape(),
                                                     &output_tensor));
    auto output = output_tensor->flat<tstring>();

    Status status;
    std::string encoded;
    for (int64_t i = 0; i < output.size(); ++i) {
      auto* wrap = handle(i).get<EntropyEncoderVariant>();
      OP_REQUIRES(context, wrap != nullptr && wrap->encoder != nullptr,
                  errors::InvalidArgument("'handle' is not an encoder"));
      status.Update(wrap->encoder->Finalize(&encoded));
      output(i) = encoded;
      handle(i).clear();
    }
    OP_REQUIRES_OK(context, status);
  }
};

REGISTER_KERNEL_BUILDER(
    Name("EntropyEncodeFinalize").Device(tensorflow::DEVICE_CPU),
    EntropyEncodeFinalizeOp);

// RangeDecoder ops -----------------------------------------------------------
class CreateRangeDecoderOp : public tensorflow::OpKernel {
 public:
  using OpKernel::OpKernel;

  void Compute(tensorflow::OpKernelContext* context) override {
    const Tensor& encoded_tensor = context->input(0);
    auto encoded = encoded_tensor.flat<tstring>();

    Tensor* output_tensor;
    OP_REQUIRES_OK(context, context->allocate_output(0, encoded_tensor.shape(),
                                                     &output_tensor));

    const Tensor& lookup = context->input(1);
    auto output = output_tensor->flat<Variant>();
    for (int64_t i = 0; i < output.size(); ++i) {
      EntropyDecoderVariant wrap;
      OP_REQUIRES_OK(context, RangeDecoderInterface::MakeShared(
                                  encoded(i), lookup, &wrap.decoder));
      wrap.holder = encoded_tensor;
      output(i) = std::move(wrap);
    }
  }
};

REGISTER_KERNEL_BUILDER(
    Name("CreateRangeDecoder").Device(tensorflow::DEVICE_CPU),
    CreateRangeDecoderOp);

class EntropyDecodeChannelOp : public tensorflow::OpKernel {
 public:
  using OpKernel::OpKernel;

  tensorflow::Status CheckShapes(tensorflow::OpKernelContext* context,
                                 TensorShape* output_shape) const {
    TensorShape suffix_shape;
    TF_RETURN_IF_ERROR(
        tensorflow::tensor::MakeShape(context->input(1), &suffix_shape));
    *output_shape = context->input(0).shape();
    output_shape->AppendShape(suffix_shape);
    return tensorflow::Status::OK();
  }

  void Compute(tensorflow::OpKernelContext* context) override {
    TensorShape output_shape;
    OP_REQUIRES_OK(context, CheckShapes(context, &output_shape));

    // This is an unnecessary shallow copy but helps avoiding a const_cast.
    Tensor handle_tensor = context->input(0);
    auto handle = handle_tensor.flat<Variant>();

    const int prefix_dims = handle_tensor.dims();

    Tensor* output_tensor;
    OP_REQUIRES_OK(context,
                   context->allocate_output(1, output_shape, &output_tensor));
    auto output =
        output_tensor->flat_inner_outer_dims<int32_t, 2>(prefix_dims - 1);

    const int64_t index_stride =
        output_shape.dims() == prefix_dims
            ? 1
            : output_shape.dim_size(output_shape.dims() - 1);
    CHECK_EQ(output.dimension(1) % index_stride, 0);

    const int64_t cost_per_unit = 80 * output.dimension(1);
    tensorflow::thread::ThreadPool* workers =
        context->device()->tensorflow_cpu_worker_threads()->workers;
    tensorflow::mutex mu;
    workers->ParallelFor(handle.size(), cost_per_unit,
                         [&handle, &mu, context, index_stride, &output](
                             int64 start, int64 limit) {
                           PerShard(handle, index_stride, output, context, &mu,
                                    start, limit);
                         });

    context->set_output(0, handle_tensor);
  }

 private:
  static void PerShard(TTypes<Variant>::Flat handle, const int64_t index_stride,
                       TTypes<int32_t>::Matrix output,
                       tensorflow::OpKernelContext* context,
                       tensorflow::mutex* mu, int64_t start, int64_t limit) {
#define REQUIRES_OK(status)                             \
  if (auto s = (status); ABSL_PREDICT_FALSE(!s.ok())) { \
    tensorflow::mutex_lock lock(*mu);                   \
    context->SetStatus(s);                              \
    return;                                             \
  }

#define REQUIRES(cond, status)        \
  if (!ABSL_PREDICT_TRUE(cond)) {     \
    tensorflow::mutex_lock lock(*mu); \
    context->SetStatus(status);       \
    return;                           \
  }

    const int64_t num_elements = output.dimension(1);
    auto* p_output = &output(start, 0);
    int64_t index = 0;

    for (int64_t i = start; i < limit; ++i) {
      auto* wrap = handle(i).get<EntropyDecoderVariant>();
      REQUIRES(wrap != nullptr && wrap->decoder != nullptr,
               errors::InvalidArgument("'handle' is not a decoder"));
      auto* decoder = wrap->decoder.get();

      for (int64_t j = 0; j < num_elements; ++j) {
        REQUIRES_OK(decoder->Decode(index++, p_output++));
        if (index == index_stride) index = 0;
      }
    }
  }

#undef REQUIRES
#undef REQUIRES_OK
};

REGISTER_KERNEL_BUILDER(
    Name("EntropyDecodeChannel").Device(tensorflow::DEVICE_CPU),
    EntropyDecodeChannelOp);

class EntropyDecodeIndexOp : public tensorflow::OpKernel {
 public:
  using OpKernel::OpKernel;

  tensorflow::Status CheckShapes(tensorflow::OpKernelContext* context,
                                 TensorShape* output_shape) const {
    TensorShape suffix_shape;
    TF_RETURN_IF_ERROR(
        tensorflow::tensor::MakeShape(context->input(2), &suffix_shape));

    *output_shape = context->input(0).shape();
    output_shape->AppendShape(suffix_shape);

    TensorShape index_shape = context->input(1).shape();
    if (index_shape != *output_shape) {
      return errors::InvalidArgument(
          "'index' shape should match 'handle' shape + 'shape': index.shape=",
          index_shape, ", handle.shape=", context->input(0).shape(),
          ", shape=", suffix_shape);
    }

    return tensorflow::Status::OK();
  }

  void Compute(tensorflow::OpKernelContext* context) override {
    TensorShape output_shape;
    OP_REQUIRES_OK(context, CheckShapes(context, &output_shape));

    // This is an unnecessary shallow copy but helps avoiding a const_cast.
    Tensor handle_tensor = context->input(0);
    auto handle = handle_tensor.flat<Variant>();

    const int prefix_dims = handle_tensor.dims();
    auto index =
        context->input(1).flat_inner_outer_dims<int32_t, 2>(prefix_dims - 1);

    CHECK_EQ(handle.dimension(0), index.dimension(0));

    Tensor* output_tensor;
    OP_REQUIRES_OK(context,
                   context->allocate_output(1, output_shape, &output_tensor));
    auto output =
        output_tensor->flat_inner_outer_dims<int32_t, 2>(prefix_dims - 1);

    const int64_t cost_per_unit = 80 * index.dimension(1);
    tensorflow::thread::ThreadPool* workers =
        context->device()->tensorflow_cpu_worker_threads()->workers;
    tensorflow::mutex mu;
    workers->ParallelFor(
        handle.size(), cost_per_unit,
        [&handle, &mu, context, index, &output](int64 start, int64 limit) {
          PerShard(handle, index, output, context, &mu, start, limit);
        });

    context->set_output(0, handle_tensor);
  }

 private:
  static void PerShard(TTypes<Variant>::Flat handle,
                       TTypes<int32_t>::ConstMatrix index,
                       TTypes<int32_t>::Matrix output,
                       tensorflow::OpKernelContext* context,
                       tensorflow::mutex* mu, int64_t start, int64_t limit) {
#define REQUIRES_OK(status)                             \
  if (auto s = (status); ABSL_PREDICT_FALSE(!s.ok())) { \
    tensorflow::mutex_lock lock(*mu);                   \
    context->SetStatus(s);                              \
    return;                                             \
  }

#define REQUIRES(cond, status)        \
  if (!ABSL_PREDICT_TRUE(cond)) {     \
    tensorflow::mutex_lock lock(*mu); \
    context->SetStatus(status);       \
    return;                           \
  }

    const int64_t num_elements = output.dimension(1);
    const int32_t* p_index = &index(start, 0);
    int32_t* p_output = &output(start, 0);

    for (int64_t i = start; i < limit; ++i) {
      auto* wrap = handle(i).get<EntropyDecoderVariant>();
      REQUIRES(wrap != nullptr && wrap->decoder != nullptr,
               errors::InvalidArgument("'handle' is not a decoder"));
      auto* decoder = wrap->decoder.get();

      for (int64_t j = 0; j < num_elements; ++j) {
        REQUIRES_OK(decoder->Decode(*(p_index++), p_output++));
      }
    }
  }

#undef REQUIRES
#undef REQUIRES_OK
};

REGISTER_KERNEL_BUILDER(
    Name("EntropyDecodeIndex").Device(tensorflow::DEVICE_CPU),
    EntropyDecodeIndexOp);

class EntropyDecodeFinalizeOp : public tensorflow::OpKernel {
 public:
  using OpKernel::OpKernel;

  void Compute(tensorflow::OpKernelContext* context) override {
    // This is an unnecessary shallow copy but helps avoiding a const_cast.
    Tensor handle_tensor = context->input(0);
    auto handle = handle_tensor.flat<Variant>();

    Tensor* output_tensor;
    OP_REQUIRES_OK(context, context->allocate_output(0, handle_tensor.shape(),
                                                     &output_tensor));
    auto output = output_tensor->flat<bool>();

    for (int64_t i = 0; i < handle.size(); ++i) {
      auto* wrap = handle(i).get<EntropyDecoderVariant>();
      OP_REQUIRES(context, wrap != nullptr && wrap->decoder != nullptr,
                  errors::InvalidArgument("'handle' is not a decoder"));
      auto status = wrap->decoder->Finalize();
      output(i) = status.ok();
      if (!status.ok()) {
        VLOG(0) << status.error_message();
      }
      handle(i).clear();
    }
  }
};

REGISTER_KERNEL_BUILDER(
    Name("EntropyDecodeFinalize").Device(tensorflow::DEVICE_CPU),
    EntropyDecodeFinalizeOp);

}  // namespace
}  // namespace tensorflow_compression
