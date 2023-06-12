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

#include "tensorflow_compression/cc/tflite/range_coder_kernels.h"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <memory>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include "tensorflow_compression/cc/lib/range_coder.h"
#include "tensorflow/lite/c/c_api.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/string_util.h"
#include "tensorflow/lite/util.h"

namespace tensorflow_compression {
namespace {

using tflite::GetInput;
using tflite::GetOutput;
using tflite::GetTensorData;
using tflite::NumDimensions;
using tflite::NumElements;
using tflite::SizeOfDimension;

#define ENSURE(expr) TF_LITE_ENSURE(context, (expr))
#define ENSURE_OK(expr) TF_LITE_ENSURE_OK(context, (expr))

int64_t ReduceProd(const int* data, int size) {
  return std::accumulate(data, data + size, int64_t{1},
                         std::multiplies<int64_t>());
}

template <TfLiteType t0, TfLiteType... t1>
inline TfLiteStatus CheckInputTypesImpl(TfLiteContext* context,
                                        TfLiteNode* node, int n) {
  const TfLiteTensor* input = GetInput(context, node, n);
  ENSURE(input != nullptr);
  ENSURE(input->type == t0);
  if constexpr (sizeof...(t1) > 0) {
    return CheckInputTypesImpl<t1...>(context, node, n + 1);
  } else {
    return kTfLiteOk;
  }
}

// REQUIRES: tflite::NumInputs(node) > 0
template <TfLiteType... t>
TfLiteStatus CheckInputTypes(TfLiteContext* context, TfLiteNode* node) {
  ENSURE(tflite::NumInputs(node) == sizeof...(t));
  return CheckInputTypesImpl<t...>(context, node, 0);
}

template <TfLiteType t0, TfLiteType... t1>
inline TfLiteStatus CheckOutputTypesImpl(TfLiteContext* context,
                                         TfLiteNode* node, int n) {
  const TfLiteTensor* output = GetOutput(context, node, n);
  ENSURE(output != nullptr);
  ENSURE(output->type == t0);
  if constexpr (sizeof...(t1) > 0) {
    return CheckOutputTypesImpl<t1...>(context, node, n + 1);
  } else {
    return kTfLiteOk;
  }
}

// REQUIRES: tflite::NumOutputs(node) > 0
template <TfLiteType... t>
TfLiteStatus CheckOutputTypes(TfLiteContext* context, TfLiteNode* node) {
  ENSURE(tflite::NumOutputs(node) == sizeof...(t));
  return CheckOutputTypesImpl<t...>(context, node, 0);
}

struct TfLiteIntArrayDeleter {
  void operator()(TfLiteIntArray* p) {
    if (p) TfLiteIntArrayFree(p);
  }
};

// REQUIRES: tensor->type == kTfLiteInt32
// REQUIRES: NumDimensions(tensor) == 1
std::unique_ptr<TfLiteIntArray, TfLiteIntArrayDeleter> MakeShape(
    const TfLiteTensor* tensor) {
  const int dims = SizeOfDimension(tensor, 0);
  std::unique_ptr<TfLiteIntArray, TfLiteIntArrayDeleter> shape(
      TfLiteIntArrayCreate(dims));
  std::copy_n(GetTensorData<int32_t>(tensor), dims, shape->data);
  return shape;
}

// REQUIRES: input->type == kTfLiteVariant
// REQUIRES: output->type == kTfLiteVariant
TfLiteStatus CopyVariantTensor(TfLiteContext* context,
                               const TfLiteTensor* input,
                               TfLiteTensor* output) {
  TF_LITE_ENSURE_OK(
      context,
      context->ResizeTensor(context, output, TfLiteIntArrayCopy(input->dims)));
  ENSURE(tflite::IsDynamicTensor(output));
  TfLiteTensorRealloc(input->bytes, output);
  std::memcpy(output->data.raw, input->data.raw, input->bytes);
  return kTfLiteOk;
}

// REQUIRES: handle->type == kTfLiteVariant.
// REQUIRES: handle is not an empty tensor.
template <typename T>
TfLiteStatus GetOpData(TfLiteContext* context, const TfLiteTensor* handle,
                       T** op_data) {
  auto variant_as_string = tflite::GetString(handle, 0);
  ENSURE(variant_as_string.len == sizeof(*op_data));
  std::memcpy(op_data, variant_as_string.str, variant_as_string.len);
  return kTfLiteOk;
}

// REQUIRES: handle->type == kTfLiteVariant.
// REQUIRES: handle is not an empty tensor.
TfLiteStatus SetOpData(TfLiteContext* context, const void* op_data,
                       TfLiteTensor* handle) {
  tflite::DynamicBuffer buffer;
  buffer.AddString(reinterpret_cast<const char*>(&op_data), sizeof(op_data));

  const int64_t num_elements = tflite::NumElements(handle);
  for (int64_t i = 1; i < num_elements; ++i) {
    buffer.AddString(nullptr, 0);
  }
  buffer.WriteToTensor(handle, nullptr);
  return kTfLiteOk;
}

struct RangeEncoderOpData {
  struct RangeEncoderState {
    RangeEncoder encoder;
    std::string sink;
  };

  std::vector<RangeEncoderState> states;
  // According to TF Lite team, TfLiteTensor's are not persistent by default,
  // and their availability is not guaranteed after the op eval.
  // https://groups.google.com/a/google.com/d/msg/tflite-users/9-7pwP2knEI/joy-bMN0AgAJ
  std::unique_ptr<int32_t[]> lookup;
  int rows = 0;
  int cols = 0;

  // Clears the state. Does not deallocate any memory.
  void Clear() {
    for (auto& state : states) {
      state.encoder = RangeEncoder();
      state.sink.clear();
    }
  }
};

void* CreateRangeEncoderOpInit(TfLiteContext* context, const char* buffer,
                               size_t length) {
  return new RangeEncoderOpData;
}

void CreateRangeEncoderOpFree(TfLiteContext* context, void* init_data) {
  delete reinterpret_cast<RangeEncoderOpData*>(init_data);
}

// TODO(ssjhv): Current implementation places the encode context in the
// opdata. Investigate if variant tensor can safely use kTfLitePersistentRo or
// kTfLiteCustom allocation strategy, so that variant tensor properly contains
// the context. The main challenge at this point is the lack of destructor call
// because TF Lite is C based and calls free() to clean up tensor data. Note
// that it is not okay to call destructor at certain ops (like
// EntropyEncodeFinalize) because that will leak memory when TF Lite interpreter
// cancels graph execution because of op errors.
TfLiteStatus CreateRangeEncoderOpPrepare(TfLiteContext* context,
                                         TfLiteNode* node) {
  ENSURE_OK((CheckInputTypes<kTfLiteInt32, kTfLiteInt32>(context, node)));
  ENSURE_OK((CheckOutputTypes<kTfLiteVariant>(context, node)));

  const TfLiteTensor* shape_tensor = GetInput(context, node, 0);
  ENSURE(shape_tensor != nullptr);
  ENSURE(NumDimensions(shape_tensor) == 1);

  const TfLiteTensor* lookup = GetInput(context, node, 1);
  ENSURE(lookup != nullptr);
  ENSURE(NumDimensions(lookup) == 2);
  ENSURE(3 <= tflite::SizeOfDimension(lookup, 1));

  TfLiteTensor* handle = GetOutput(context, node, 0);
  ENSURE(handle != nullptr);

  // Begin preparation.
  auto* op_data = reinterpret_cast<RangeEncoderOpData*>(node->user_data);

  const int64_t lookup_size = NumElements(lookup);
  op_data->lookup = std::make_unique<int32_t[]>(lookup_size);
  op_data->rows = SizeOfDimension(lookup, 0);
  op_data->cols = SizeOfDimension(lookup, 1);

  ENSURE(tflite::IsConstantTensor(shape_tensor));
  auto shape = MakeShape(shape_tensor);
  const int64_t num_elements = NumElements(shape.get());
  op_data->states.resize(num_elements);

  ENSURE_OK(context->ResizeTensor(context, handle, shape.release()));
  ENSURE_OK(SetOpData(context, op_data, handle));
  return kTfLiteOk;
}

TfLiteStatus CreateRangeEncoderOpEval(TfLiteContext* context,
                                      TfLiteNode* node) {
  auto* op_data = reinterpret_cast<RangeEncoderOpData*>(node->user_data);
  op_data->Clear();

  const TfLiteTensor* lookup = GetInput(context, node, 1);
  ENSURE(lookup != nullptr);
  const int64_t lookup_size =
      static_cast<int64_t>(op_data->rows) * op_data->cols;
  std::copy_n(GetTensorData<int32_t>(lookup), lookup_size,
              op_data->lookup.get());

  return kTfLiteOk;
}

TfLiteStatus EntropyEncodeIndexOpPrepare(TfLiteContext* context,
                                         TfLiteNode* node) {
  ENSURE_OK((CheckInputTypes<kTfLiteVariant, kTfLiteInt32, kTfLiteInt32>(
      context, node)));
  ENSURE_OK((CheckOutputTypes<kTfLiteVariant>(context, node)));

  const TfLiteTensor* handle = GetInput(context, node, 0);
  ENSURE(handle != nullptr);
  const TfLiteTensor* index = GetInput(context, node, 1);
  ENSURE(index != nullptr);
  const TfLiteTensor* value = GetInput(context, node, 2);
  ENSURE(value != nullptr);

  // Check if handle.shape is a prefix of value.shape.
  ENSURE(NumDimensions(handle) <= NumDimensions(value) &&
         TfLiteIntArrayEqualsArray(handle->dims, handle->dims->size,
                                   value->dims->data));
  ENSURE(TfLiteIntArrayEqual(value->dims, index->dims));

  TfLiteTensor* alias = GetOutput(context, node, 0);
  ENSURE(alias != nullptr);
  ENSURE_OK(CopyVariantTensor(context, handle, alias));

  return kTfLiteOk;
}

TfLiteStatus EntropyEncodeIndexOpEval(TfLiteContext* context,
                                      TfLiteNode* node) {
  const TfLiteTensor* handle = GetInput(context, node, 0);
  ENSURE(handle != nullptr);
  const int prefix_dims = NumDimensions(handle);

  RangeEncoderOpData* op_data;
  ENSURE_OK(GetOpData(context, handle, &op_data));

  const TfLiteTensor* index_tensor = GetInput(context, node, 1);
  ENSURE(index_tensor != nullptr);
  const TfLiteTensor* value_tensor = GetInput(context, node, 2);
  ENSURE(value_tensor != nullptr);

  const int64_t num_elements =
      ReduceProd(value_tensor->dims->data + prefix_dims,
                 value_tensor->dims->size - prefix_dims);

  const int32_t* index = GetTensorData<int32_t>(index_tensor);
  const int32_t* value = GetTensorData<int32_t>(value_tensor);
  const int32_t* lookup = op_data->lookup.get();
  const int lookup_stride = op_data->cols;

  for (auto& state : op_data->states) {
    for (int64_t j = 0; j < num_elements; ++j, ++value, ++index) {
      DCHECK(0 <= *index && *index < op_data->rows);
      DCHECK(0 <= *value && *value + 2 < lookup_stride);
      const int32_t* row = lookup + ((*index) * lookup_stride);
      const int32_t precision = row[0];
      state.encoder.Encode(row[*value + 1], row[*value + 2], precision,
                           &state.sink);
    }
  }

  return kTfLiteOk;
}

TfLiteStatus EntropyEncodeFinalizeOpPrepare(TfLiteContext* context,
                                            TfLiteNode* node) {
  ENSURE_OK((CheckInputTypes<kTfLiteVariant>(context, node)));
  ENSURE_OK((CheckOutputTypes<kTfLiteString>(context, node)));

  const TfLiteTensor* handle = GetInput(context, node, 0);
  ENSURE(handle != nullptr);
  ENSURE(NumElements(handle) > 0);

  TfLiteTensor* encode = GetOutput(context, node, 0);
  ENSURE(encode != nullptr);
  ENSURE_OK(
      context->ResizeTensor(context, encode, TfLiteIntArrayCopy(handle->dims)));

  return kTfLiteOk;
}

TfLiteStatus EntropyEncodeFinalizeOpEval(TfLiteContext* context,
                                         TfLiteNode* node) {
  const TfLiteTensor* handle = GetInput(context, node, 0);
  ENSURE(handle != nullptr);

  RangeEncoderOpData* op_data;
  ENSURE_OK(GetOpData(context, handle, &op_data));

  tflite::DynamicBuffer buffer;
  for (auto& state : op_data->states) {
    state.encoder.Finalize(&state.sink);
    buffer.AddString(state.sink.data(), state.sink.size());
  }

  TfLiteTensor* output = GetOutput(context, node, 0);
  ENSURE(output != nullptr);
  buffer.WriteToTensor(output, nullptr);

  op_data->Clear();
  return kTfLiteOk;
}

struct RangeDecoderOpData {
  std::vector<RangeDecoder> decoders;

  std::unique_ptr<int32_t[]> lookup;
  int rows = 0;
  int cols = 0;

  std::string encode;
};

void* CreateRangeDecoderOpInit(TfLiteContext* context, const char* buffer,
                               size_t length) {
  return new RangeDecoderOpData;
}

void CreateRangeDecoderOpFree(TfLiteContext* context, void* init_data) {
  delete reinterpret_cast<RangeDecoderOpData*>(init_data);
}

TfLiteStatus CreateRangeDecoderOpPrepare(TfLiteContext* context,
                                         TfLiteNode* node) {
  ENSURE_OK((CheckInputTypes<kTfLiteString, kTfLiteInt32>(context, node)));
  ENSURE_OK((CheckOutputTypes<kTfLiteVariant>(context, node)));

  const TfLiteTensor* encode = GetInput(context, node, 0);
  ENSURE(encode != nullptr);
  ENSURE(NumElements(encode) > 0);

  const TfLiteTensor* lookup = GetInput(context, node, 1);
  ENSURE(lookup != nullptr);
  ENSURE(NumDimensions(lookup) == 2);
  ENSURE(3 <= SizeOfDimension(lookup, 1));

  TfLiteTensor* handle = GetOutput(context, node, 0);
  ENSURE(handle != nullptr);
  TF_LITE_ENSURE_OK(
      context,
      context->ResizeTensor(context, handle, TfLiteIntArrayCopy(encode->dims)));

  const int64_t num_elements = NumElements(encode);
  auto* op_data = reinterpret_cast<RangeDecoderOpData*>(node->user_data);
  op_data->decoders.resize(num_elements, RangeDecoder(""));
  op_data->encode.clear();

  const int64_t lookup_size = NumElements(lookup);
  op_data->lookup = std::make_unique<int32_t[]>(lookup_size);
  op_data->rows = SizeOfDimension(lookup, 0);
  op_data->cols = SizeOfDimension(lookup, 1);

  ENSURE_OK(SetOpData(context, op_data, handle));
  return kTfLiteOk;
}

TfLiteStatus CreateRangeDecoderOpEval(TfLiteContext* context,
                                      TfLiteNode* node) {
  const TfLiteTensor* encode = GetInput(context, node, 0);
  ENSURE(encode != nullptr);
  const int64_t num_elements = NumElements(encode);

  // Copy the string buffer into Op data.
  auto* op_data = reinterpret_cast<RangeDecoderOpData*>(node->user_data);
  op_data->encode.assign(encode->data.raw, encode->bytes);

  // Initialize the decoders.
  ENSURE(op_data->decoders.size() == num_elements);
  for (int64_t i = 0; i < num_elements; ++i) {
    // Each string element in `encode` is a substring inside the buffer pointed
    // by `encode->data`. Compute the offset of the substring, and find the
    // corresponding substring inside `op_data->encode`.
    const auto str = tflite::GetString(encode, i);
    const auto offset = str.str - encode->data.raw;
    DCHECK_LT(offset, encode->bytes);
    op_data->decoders[i] = RangeDecoder(
        absl::string_view(op_data->encode.data() + offset, str.len));
  }

  // Copy the lookup information.
  const TfLiteTensor* lookup = GetInput(context, node, 1);
  ENSURE(lookup != nullptr);
  const int64_t lookup_size =
      static_cast<int64_t>(op_data->rows) * op_data->cols;
  std::copy_n(GetTensorData<int32_t>(lookup), lookup_size,
              op_data->lookup.get());

  return kTfLiteOk;
}

TfLiteStatus EntropyDecodeIndexOpPrepare(TfLiteContext* context,
                                         TfLiteNode* node) {
  ENSURE_OK((CheckInputTypes<kTfLiteVariant, kTfLiteInt32, kTfLiteInt32>(
      context, node)));
  ENSURE_OK((CheckOutputTypes<kTfLiteVariant, kTfLiteInt32>(context, node)));

  const TfLiteTensor* handle = GetInput(context, node, 0);
  ENSURE(handle != nullptr);
  TfLiteTensor* handle_alias = GetOutput(context, node, 0);
  ENSURE(handle_alias != nullptr);
  ENSURE(NumElements(handle) > 0);

  // Forward the handle alias.
  ENSURE_OK(CopyVariantTensor(context, handle, handle_alias));

  const TfLiteTensor* index = GetInput(context, node, 1);
  ENSURE(index != nullptr);
  const TfLiteTensor* suffix_shape = GetInput(context, node, 2);
  ENSURE(suffix_shape != nullptr);

  ENSURE(tflite::IsConstantTensor(suffix_shape));
  ENSURE(NumDimensions(suffix_shape) == 1);
  const int prefix_dims = NumDimensions(handle);
  const int suffix_dims = SizeOfDimension(suffix_shape, 0);

  // index.shape should be handle.shape concatenated by suffix_dims.
  ENSURE(NumDimensions(index) == prefix_dims + suffix_dims);
  ENSURE(std::equal(index->dims->data, index->dims->data + prefix_dims,
                    handle->dims->data));
  ENSURE(std::equal(index->dims->data + prefix_dims,
                    index->dims->data + prefix_dims + suffix_dims,
                    GetTensorData<int32_t>(suffix_shape)));

  TfLiteTensor* decode = GetOutput(context, node, 1);
  ENSURE(decode != nullptr);
  ENSURE_OK(
      context->ResizeTensor(context, decode, TfLiteIntArrayCopy(index->dims)));

  return kTfLiteOk;
}

TfLiteStatus EntropyDecodeIndexOpEval(TfLiteContext* context,
                                      TfLiteNode* node) {
  const TfLiteTensor* handle = GetInput(context, node, 0);
  ENSURE(handle != nullptr);
  const int prefix_dims = NumDimensions(handle);

  RangeDecoderOpData* op_data;
  ENSURE_OK(GetOpData(context, handle, &op_data));

  TfLiteTensor* decode_tensor = GetOutput(context, node, 1);
  ENSURE(decode_tensor != nullptr);
  const TfLiteTensor* index_tensor = GetInput(context, node, 1);
  ENSURE(index_tensor != nullptr);

  const int64_t num_elements =
      ReduceProd(decode_tensor->dims->data + prefix_dims,
                 decode_tensor->dims->size - prefix_dims);

  int32_t* decode = GetTensorData<int32_t>(decode_tensor);
  const int32_t* index = GetTensorData<int32_t>(index_tensor);
  const int32_t* lookup = op_data->lookup.get();
  const int lookup_stride = op_data->cols;

  for (auto& decoder : op_data->decoders) {
    for (int64_t j = 0; j < num_elements; ++j, ++decode, ++index) {
      DCHECK(0 <= *index && *index < op_data->rows);
      const int32_t* row = lookup + ((*index) * lookup_stride);
      const int32_t precision = *row;
      *decode =
          decoder.Decode(absl::MakeSpan(row + 1, lookup_stride - 1), precision);
    }
  }

  return kTfLiteOk;
}

TfLiteStatus EntropyDecodeFinalizeOpPrepare(TfLiteContext* context,
                                            TfLiteNode* node) {
  ENSURE_OK((CheckInputTypes<kTfLiteVariant>(context, node)));
  ENSURE_OK((CheckOutputTypes<kTfLiteBool>(context, node)));

  const TfLiteTensor* handle = GetInput(context, node, 0);
  ENSURE(handle != nullptr);
  ENSURE(NumElements(handle) > 0);

  TfLiteTensor* success = GetOutput(context, node, 0);
  ENSURE(success != nullptr);
  ENSURE_OK(context->ResizeTensor(context, success,
                                  TfLiteIntArrayCopy(handle->dims)));
  return kTfLiteOk;
}

TfLiteStatus EntropyDecodeFinalizeOpEval(TfLiteContext* context,
                                         TfLiteNode* node) {
  const TfLiteTensor* handle = GetInput(context, node, 0);
  ENSURE(handle != nullptr);

  RangeDecoderOpData* op_data;
  ENSURE_OK(GetOpData(context, handle, &op_data));

  TfLiteTensor* success_tensor = GetOutput(context, node, 0);
  ENSURE(success_tensor != nullptr);
  bool* success = GetTensorData<bool>(success_tensor);
  for (auto& decoder : op_data->decoders) {
    *success++ = decoder.Finalize();
    decoder = RangeDecoder("");
  }
  return kTfLiteOk;
}

#undef ENSURE_OK
#undef ENSURE

}  // namespace

void RegisterRangeCoderOps(tflite::ops::builtin::BuiltinOpResolver* resolver) {
  {
    static TfLiteRegistration r = {
        CreateRangeEncoderOpInit, CreateRangeEncoderOpFree,
        CreateRangeEncoderOpPrepare, CreateRangeEncoderOpEval};
    resolver->AddCustom("CreateRangeEncoder", &r);
  }

  {
    static TfLiteRegistration r = {nullptr, nullptr,
                                   EntropyEncodeIndexOpPrepare,
                                   EntropyEncodeIndexOpEval};
    resolver->AddCustom("EntropyEncodeIndex", &r);
  }

  {
    static TfLiteRegistration r = {nullptr, nullptr,
                                   EntropyEncodeFinalizeOpPrepare,
                                   EntropyEncodeFinalizeOpEval};
    resolver->AddCustom("EntropyEncodeFinalize", &r);
  }

  {
    static TfLiteRegistration r = {
        CreateRangeDecoderOpInit, CreateRangeDecoderOpFree,
        CreateRangeDecoderOpPrepare, CreateRangeDecoderOpEval};
    resolver->AddCustom("CreateRangeDecoder", &r);
  }

  {
    static TfLiteRegistration r = {nullptr, nullptr,
                                   EntropyDecodeIndexOpPrepare,
                                   EntropyDecodeIndexOpEval};
    resolver->AddCustom("EntropyDecodeIndex", &r);
  }

  {
    static TfLiteRegistration r = {nullptr, nullptr,
                                   EntropyDecodeFinalizeOpPrepare,
                                   EntropyDecodeFinalizeOpEval};
    resolver->AddCustom("EntropyDecodeFinalize", &r);
  }
}

}  // namespace tensorflow_compression
