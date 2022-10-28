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
#include "tensorflow_compression/cc/lib/bit_coder.h"

#include <stdlib.h>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <limits>

#include "absl/base/internal/endian.h"
#include "absl/status/status.h"

namespace tensorflow_compression {

BitWriter::BitWriter()
    : next_index_(0),
      bits_in_buffer_(0),
      buffer_(0) {}

void BitWriter::WriteBits(uint32_t count, uint64_t bits) {
  assert(count <= kMaxBitsPerCall);
  // This implementation assumes unwritten MSBs in the buffer are zero.
  // Clear the unused MSBs in bits first, then "or" it into the buffer.
  bits &= (uint64_t{1} << count) - 1;
  buffer_ |= bits << bits_in_buffer_;
  bits_in_buffer_ += count;
  // TODO(jonycgn): Investigate performance of buffer resizing.
  data_.resize(next_index_ + 8);
  absl::little_endian::Store64(&data_[next_index_], buffer_);
  size_t bytes_in_buffer = bits_in_buffer_ / 8;
  bits_in_buffer_ -= bytes_in_buffer * 8;
  buffer_ >>= bytes_in_buffer * 8;
  next_index_ += bytes_in_buffer;
}

void BitWriter::WriteOneBit(uint64_t bit) { return WriteBits(1, bit); }

void BitWriter::WriteGamma(int32_t value) {
  assert(value > 0);
  auto bit_width = absl::bit_width(static_cast<uint32_t>(value));
  // Encode most significant bit with a unary code.
  WriteBits(bit_width - 1, 0);
  WriteBits(1, 1);
  // Encode least significant bits with a binary code.
  WriteBits(bit_width - 1, value);
}

void BitWriter::WriteRice(int32_t value, int parameter) {
  assert(value >= 0);
  assert(parameter >= 0);
  // Encode most significant bits with a unary code.
  uint32_t num_zeros = value >> parameter;
  while (num_zeros > kMaxBitsPerCall) {
    WriteBits(kMaxBitsPerCall, 0);
    num_zeros -= kMaxBitsPerCall;
  }
  WriteBits(num_zeros, 0);
  WriteBits(1, 1);
  // Encode least significant bits with a binary code.
  WriteBits(parameter, value);
}

absl::string_view BitWriter::GetData() {
  size_t num_bytes = next_index_;
  if (bits_in_buffer_) {
    assert(bits_in_buffer_ < 8);
    ++num_bytes;
  }
  return absl::string_view(data_.data(), num_bytes);
}

// bytes need not be aligned nor padded!
BitReader::BitReader(const absl::string_view data)
    : next_byte_(data.data()),
      end_byte_(data.data() + data.size()),
      bits_in_buffer_(0),
      buffer_(0) {}

void BitReader::Refill() {
  const ptrdiff_t bytes_remaining = end_byte_ - next_byte_;
  if (bytes_remaining < 8) {
    assert(bytes_remaining >= 0);
    const size_t bytes_to_copy = std::min((63 - bits_in_buffer_) / 8,
                                          static_cast<size_t>(bytes_remaining));
    // Do not attempt to dereference pointer if there are no bytes available
    // (empty bit string, end of bit string).
    if (!bytes_to_copy) return;
    uint64_t x = 0;
    memcpy(&x, next_byte_, bytes_to_copy);
    buffer_ |= absl::little_endian::ToHost(x) << bits_in_buffer_;
    next_byte_ += bytes_to_copy;
    bits_in_buffer_ += bytes_to_copy * 8;
    assert(bits_in_buffer_ < 64);
  } else {
    // It's safe to load 64 bits; insert valid (possibly nonzero) bits above
    // bits_in_buffer_. The shift requires bits_in_buffer_ < 64.
    buffer_ |= absl::little_endian::Load64(next_byte_) << bits_in_buffer_;
    // Advance by bytes fully absorbed into the buffer.
    next_byte_ += (63 - bits_in_buffer_) / 8;
    // We absorbed a multiple of 8 bits, so the lower 3 bits of bits_in_buffer_
    // must remain unchanged, otherwise the next refill's shifted bits will
    // not align with buffer_. Set the three upper bits so the result >= 56.
    bits_in_buffer_ = 56 + (bits_in_buffer_ % 8);
    assert(56 <= bits_in_buffer_ && bits_in_buffer_ < 64);
  }
}

absl::StatusOr<uint64_t> BitReader::ReadBits(size_t count) {
  Refill();
  if (bits_in_buffer_ < count) {
    return absl::DataLossError("Out of bits to read.");
  }
  const uint64_t mask = (1ULL << count) - 1;
  const uint64_t bits = buffer_ & mask;
  bits_in_buffer_ -= count;
  buffer_ >>= count;
  return bits;
}

absl::StatusOr<uint64_t> BitReader::ReadOneBit() { return ReadBits(1); }

absl::StatusOr<int32_t> BitReader::ReadGamma() {
  // Decode most significant bit with a unary code.
  int32_t bit_width = 1;
  while (true) {
    auto bit = ReadOneBit();
    if (!bit.ok()) return bit;
    if (*bit) break;
    ++bit_width;
  }
  if (bit_width > 31) {
    return absl::DataLossError("Exceeded maximum gamma bit width.");
  }
  int32_t msb = 1 << (bit_width - 1);
  // Decode least significant bits with a binary code.
  auto lsbs = ReadBits(bit_width - 1);
  if (!lsbs.ok()) return lsbs;
  return msb | *lsbs;
}

absl::StatusOr<int32_t> BitReader::ReadRice(int parameter) {
  assert(parameter >= 0);
  // Decode most significant bits with a unary code.
  int32_t msbs = 0;
  while (true) {
    auto bit = ReadOneBit();
    if (!bit.ok()) return bit;
    if (*bit) break;
    ++msbs;
  }
  // Decode least significant bits with a binary code.
  auto lsbs = ReadBits(parameter);
  if (!lsbs.ok()) return lsbs;
  return (msbs << parameter) | *lsbs;
}

}  // namespace tensorflow_compression
