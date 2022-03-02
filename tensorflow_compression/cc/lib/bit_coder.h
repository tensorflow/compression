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
#ifndef THIRD_PARTY_TENSORFLOW_COMPRESSION_CC_LIB_BIT_CODER_H_
#define THIRD_PARTY_TENSORFLOW_COMPRESSION_CC_LIB_BIT_CODER_H_

#include <cstdint>
#include <limits>
#include <string>
#include <type_traits>

#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "tensorflow/core/platform/logging.h"

// TODO(nicolemitchell) inline commonly used methods for performance reasons.
namespace tensorflow_compression {

class BitWriter {
 public:
  BitWriter() = default;
  void Allocate(size_t maximum_bit_size);
  void WriteOneBit(uint64_t bit);
  void WriteGamma(uint32_t value);
  void ZeroPadToByte();
  char* GetData();
  size_t GetBytesWritten();

  // Encoding an int32_t value requires max 64 bits.
  static constexpr int32_t kMaxGammaBits = 64;
  static constexpr size_t kMaxBitsPerCall = 57;


 private:
  void WriteBits(uint32_t count, uint64_t bits);

  std::unique_ptr<char[]> data_;
  size_t bytes_written_ = 0;
  size_t bits_in_buffer_ = 0;
  uint64_t buffer_ = 0;
};

class BitReader {
 public:
  BitReader(const absl::string_view bytes);
  void Refill();
  uint64_t ReadOneBit();
  uint32_t ReadGamma();
  size_t TotalBitsConsumed() const;
  size_t TotalBytes() const;
  absl::Status Close();

  static constexpr size_t kMaxBitsPerCall = 56;

 private:
  uint64_t ReadBits(size_t nbits);

  uint64_t buf_;
  size_t bits_in_buf_;  // [0, 64)
  const char* next_byte_;
  const char* end_minus_8_;  // for refill bounds check
  const char* first_byte_;
  uint64_t bits_consumed_{0};
  bool close_called_{false};
};
}  // namespace tensorflow_compression

#endif  // THIRD_PARTY_TENSORFLOW_COMPRESSION_CC_LIB_BIT_CODER_H_
