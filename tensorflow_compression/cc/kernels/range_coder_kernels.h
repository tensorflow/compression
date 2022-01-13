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

#ifndef TENSORFLOW_COMPRESSION_CC_KERNELS_RANGE_CODER_KERNELS_H_
#define TENSORFLOW_COMPRESSION_CC_KERNELS_RANGE_CODER_KERNELS_H_

#include <cstdint>
#include <string>

#include "absl/base/integral_types.h"
#include "absl/types/span.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow_compression {

class EntropyEncoderInterface {
 public:
  virtual ~EntropyEncoderInterface() = default;
  virtual tensorflow::Status Encode(int32_t index, int32_t value) = 0;
  virtual tensorflow::Status Finalize(std::string* sink) = 0;
};

class EntropyDecoderInterface {
 public:
  virtual ~EntropyDecoderInterface() = default;
  virtual tensorflow::Status Decode(int32_t index, int32_t* output) = 0;
  virtual tensorflow::Status Finalize() = 0;
};

}  // namespace tensorflow_compression

#endif  // TENSORFLOW_COMPRESSION_CC_KERNELS_RANGE_CODER_KERNELS_H_
