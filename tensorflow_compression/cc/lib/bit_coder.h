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

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"

// TODO(nicolemitchell) inline commonly used methods for performance reasons.
namespace tensorflow_compression {

class BitWriter {
 public:
  BitWriter();
  void WriteBits(uint32_t count, uint64_t bits);
  void WriteOneBit(uint64_t bit);
  void WriteGamma(int32_t value);
  void WriteRice(int32_t value, const int parameter);
  absl::string_view GetData();

  // WriteGamma() encodes integers to 2n - 1 bits, where n is the bit width of
  // the integer. For int32_t (> 0), n <= 31.
  static constexpr size_t kMaxGammaBits = 61;
  // After each WriteBits(), the buffer contains a maximum of 7 bits. So we can
  // safely put 57 more bits to fill the buffer.
  static constexpr size_t kMaxBitsPerCall = 57;

 private:
  std::string data_;
  std::string::size_type next_index_;
  size_t bits_in_buffer_;
  uint64_t buffer_;
};

class BitReader {
 public:
  BitReader(const absl::string_view data);
  absl::StatusOr<uint64_t> ReadBits(size_t count);
  absl::StatusOr<uint64_t> ReadOneBit();
  absl::StatusOr<int32_t> ReadGamma();
  absl::StatusOr<int32_t> ReadRice(const int parameter);

 private:
  void Refill();

  const char* next_byte_;
  const char* end_byte_;
  size_t bits_in_buffer_;
  uint64_t buffer_;
};
}  // namespace tensorflow_compression

#endif  // THIRD_PARTY_TENSORFLOW_COMPRESSION_CC_LIB_BIT_CODER_H_
