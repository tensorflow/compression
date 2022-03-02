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

#include <cstdint>

#include "absl/base/internal/endian.h"
#include "absl/status/status.h"
#include "tensorflow/core/platform/logging.h"

namespace tensorflow_compression {

void BitWriter::Allocate(size_t maximum_bit_size) {
  assert(data_ == nullptr);
  // We write 8 bytes at a time, so we might over-write by 8 bytes.
  data_ = std::make_unique<char[]>(maximum_bit_size / 8 + 32);  // Slightly more
}

void BitWriter::WriteBits(uint32_t count, uint64_t bits) {
  DCHECK_LE(count, kMaxBitsPerCall);  // Max remaining room in buffer.
  buffer_ |= bits << bits_in_buffer_;
  bits_in_buffer_ += count;
  absl::little_endian::Store64(GetData() + bytes_written_, buffer_);
  size_t bytes_in_buffer = bits_in_buffer_ / 8;
  bits_in_buffer_ -= bytes_in_buffer * 8;
  buffer_ >>= bytes_in_buffer * 8;
  bytes_written_ += bytes_in_buffer;
}

void BitWriter::WriteOneBit(uint64_t bit) {
  return WriteBits(1, bit);
}

void BitWriter::WriteGamma(uint32_t value) {
  DCHECK_GT(value, 0);
  int32_t n = absl::bit_width(value);
  WriteBits(n - 1, 0);
  // Store most significant bit first.
  WriteBits(1, 1);
  WriteBits(n - 1, value - (1 << (n - 1)));
}

void BitWriter::ZeroPadToByte() {
  if (bits_in_buffer_ != 0) {
    WriteBits(8 - bits_in_buffer_, 0);
  }
}

char* BitWriter::GetData() { return data_.get(); }

size_t BitWriter::GetBytesWritten() { return bytes_written_; }

// bytes need not be aligned nor padded!
BitReader::BitReader(const absl::string_view bytes)
    : buf_(0),
      bits_in_buf_(0),
      next_byte_(bytes.data()),
      // Assumes first_byte_ >= 8.
      end_minus_8_(bytes.data() - 8 + bytes.size()),
      first_byte_(bytes.data()) {
  CHECK_GT(bytes.size(), 0);
  Refill();
}

void BitReader::Refill() {
  if (next_byte_ > end_minus_8_) {
    int bytes_left_in_stream = end_minus_8_ + 8 - next_byte_;
    CHECK_GE(bytes_left_in_stream, 0);
    uint64_t x = 0;
    int bytes_to_copy = std::min(static_cast<int>((63 - bits_in_buf_) / 8),
                                 bytes_left_in_stream);
    memcpy(&x, next_byte_, bytes_to_copy);
    buf_ |= absl::little_endian::ToHost(x) << bits_in_buf_;
    next_byte_ += bytes_to_copy;
    bits_in_buf_ += bytes_to_copy * 8;
    DCHECK_LT(bits_in_buf_, 64);
  } else {
    // It's safe to load 64 bits; insert valid (possibly nonzero) bits above
    // bits_in_buf_. The shift requires bits_in_buf_ < 64.
    buf_ |= absl::little_endian::Load64(next_byte_) << bits_in_buf_;

    // Advance by bytes fully absorbed into the buffer.
    next_byte_ += (63 - bits_in_buf_) / 8;

    // We absorbed a multiple of 8 bits, so the lower 3 bits of bits_in_buf_
    // must remain unchanged, otherwise the next refill's shifted bits will
    // not align with buf_. Set the three upper bits so the result >= 56.
    bits_in_buf_ = 56 + (bits_in_buf_ % 8);
    DCHECK(56 <= bits_in_buf_ && bits_in_buf_ < 64);
  }
}

uint64_t BitReader::ReadBits(size_t nbits) {
  DCHECK(!close_called_);
  DCHECK_LE(nbits, kMaxBitsPerCall);
  Refill();
  const uint64_t mask = (1ULL << nbits) - 1;
  const uint64_t bits = buf_ & mask;
  DCHECK_GE(bits_in_buf_, nbits);
  bits_in_buf_ -= nbits;
  buf_ >>= nbits;
  bits_consumed_ += nbits;

  return bits;
}

uint64_t BitReader::ReadOneBit() {
  return ReadBits(1);
}

uint32_t BitReader::ReadGamma() {
  int32_t n = 1;  // Initializing n as 1 for consistency with WriteGamma.
  while (ReadBits(1) == 0) {
    ++n;
    DCHECK_LE(n, 32);  // Ensure we do not read past end of bitstream.
  }
  uint32_t value = 1 << (n - 1);
  return value | ReadBits(n - 1);
}

size_t BitReader::TotalBitsConsumed() const {
  return static_cast<size_t>(bits_consumed_);
}

size_t BitReader::TotalBytes() const {
  return static_cast<size_t>(end_minus_8_ + 8 - first_byte_);
}

// Close the bit reader and return whether all the previous reads were
// successful. Close must be called once.
absl::Status BitReader::Close() {
  DCHECK(!close_called_);
  close_called_ = true;
  if (!first_byte_) return absl::OkStatus();
  if (TotalBitsConsumed() > TotalBytes() * 8) {
    return absl::OutOfRangeError(
        "Read more bits than available in the bit_reader");
  }
  return absl::OkStatus();
}
}  // namespace tensorflow_compression
