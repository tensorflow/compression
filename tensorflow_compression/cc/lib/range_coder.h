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

#ifndef TENSORFLOW_COMPRESSION_CC_LIB_RANGE_CODER_H_
#define TENSORFLOW_COMPRESSION_CC_LIB_RANGE_CODER_H_

#include <cstdint>
#include <limits>
#include <string>
#include <type_traits>

#include "absl/base/integral_types.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "tensorflow/core/platform/logging.h"

namespace tensorflow_compression {
class RangeEncoder {
 public:
  RangeEncoder() = default;

  // Encodes a half-open interval [lower / 2^precision, upper / 2^precision).
  // Suppose that each character to be encoded is from an integer-valued
  // distribution. When encoding a random character x0, the arguments lower and
  // upper represent
  //
  //   Pr(X < x0) = lower / 2^precision,
  //   Pr(X <= x0) = upper / 2^precision,
  //
  // where X is a random variable following the distribution. Note that
  // `precision` determines the granularity of probability masses.
  //
  // For example, assume that the distribution has possible outputs 0, 1, 2, ...
  // To encode value 0, lower = 0 and upper = Pr(X = 0).
  // To encode value 1, lower = Pr(X = 0) and upper = Pr(X = 0 or 1).
  // To encode value 2, lower = Pr(X = 0 or 1) and upper = Pr(X = 0, 1, or 2).
  // ...
  //
  // REQUIRES: 0 <= lower < upper <= 2^precision.
  // REQUIRES: 0 < precision <= 16.
  void Encode(int32_t lower, int32_t upper, int precision, std::string* sink);

  // The encode may contain some under-determined values from previous encoding.
  // After Encode() calls, Finalize() must be called. Otherwise the encoded
  // string may not be decoded.
  void Finalize(std::string* sink) const;

  // Encode() does not check for incorrect arguments due to performance reasons,
  // but it may be challenging for debug purposes. This function may be helpful
  // during development phase for catching incorrect usage.
  absl::Status CheckForError(int32_t lower, int32_t upper, int precision) const;

 private:
  uint32_t base_ = 0;
  uint32_t size_minus1_ = std::numeric_limits<uint32_t>::max();
  uint64_t delay_ = 0;
};

class RangeDecoder {
 public:
  // Holds a reference to `source`. The caller has to make sure that `source`
  // outlives the decoder object.
  //
  // REQUIRES: `precision` must be the same as the encoder's precision.
  // REQUIRES: 0 < precision <= 16.
  explicit RangeDecoder(absl::string_view source)
      : current_(source.begin()), end_(source.end()) {
    Read16BitValue();
    Read16BitValue();
  }

  // Decodes a character from `source` using CDF. The size of `cdf` should be
  // one more than the number of the character in the alphabet.
  //
  // If x0, x1, x2, ... are the possible characters (in increasing order) from
  // the distribution, then
  //   cdf[0] = 0
  //   cdf[1] = Pr(X <= x0),
  //   cdf[2] = Pr(X <= x1),
  //   cdf[3] = Pr(X <= x2),
  //   ...
  //
  // The returned value is an index to `cdf` where the decoded character
  // corresponds to.
  //
  // REQUIRES: cdf.size() > 1.
  // REQUIRES: cdf[i] <= cdf[i + 1] for i = 0, 1, ..., cdf.size() - 2.
  // REQUIRES: cdf[0] == 0.
  // REQUIRES: cdf[cdf.size() - 1] <= 2^precision.
  // REQUIRES: 0 < precision <= 16.
  //
  // In practice the last element of `cdf` should equal to 2^precision.
  int Decode(absl::Span<const int16_t> cdf, int precision) {
    return DecodeInternal<BinarySearch>(cdf, precision);
  }
  int Decode(absl::Span<const int32_t> cdf, int precision) {
    return DecodeInternal<BinarySearch>(cdf, precision);
  }

  // Decode() uses binary search internally. Use this variant when it makes
  // sense to use linear search instead.
  //
  // For instance,
  //   - when cdf.size() is small,
  //
  //   OR
  //
  //   - when cdf has low entropy, i.e., there are few symbols have high
  //   likelihoods, and when symbols are sorted in descending order of their
  //   likelihoods.
  int DecodeLinearly(absl::Span<const int16_t> cdf, int precision) {
    return DecodeInternal<LinearSearch>(cdf, precision);
  }
  int DecodeLinearly(absl::Span<const int32_t> cdf, int precision) {
    return DecodeInternal<LinearSearch>(cdf, precision);
  }

  // When called after decoding all the values, this function returns a sanity
  // check if the decoding was successful, by examining the decoder state. The
  // returned value may be a false-positive, i.e., the decoding may have had
  // errors but the returned value may be True. However, there are no
  // false alarms, i.e., if the returned value is False then there was a
  // decoding error, e.g., decoding not enough characters from the stream.
  //
  // WARNING: This is a weak sanity check, especially easy to fail when user
  // decoded more values than contained in the encoded stream.
  //
  // TODO(ssjhv): Consider removing implicit trailing zeros in
  // RangeEncoder::Finalize(). That feature is one of the main reasons decoder
  // cannot detect decode-too-many errors.
  bool Finalize() const {
    // If decoder did not read to the end, return false.
    if (current_ != end_) {
      return false;
    }

    const uint32_t upper = base_ + size_minus1_;

    // If the encoder finalized in state 0 and base == 0, then encoder picks
    // implicit zeros for the last four bytes. RangeEncoder does not write any
    // more byte in its Finalize(). In this case, value_ == 0.
    //
    // If the encoder finalized in state 1, then value_ == 0. State 1 can be
    // detected by checking whether the largest element in the interval, i.e.,
    // base + (size - 1), overflows or not. This corresponds to delay_ != 0 in
    // RangeEncoder::Finalize().
    if (base_ == 0 || upper < base_) {
      return value_ == 0;
    }

    // This is when the encoder finalized in state 0 and base != 0. The encoder
    // rounds up base to the next multiple of 2^24 or 2^16.
    const int shift = (((base_ - 1) >> 24) < (upper >> 24)) ? 24 : 16;
    const uint32_t mid = ((base_ - 1) >> shift) + 1;
    return (mid << shift) == value_;
  }

  // Decode() does not check for incorrect arguments due to performance reasons,
  // but it may be challenging for debug purposes. This function may be helpful
  // during development phase for catch incorrect usage.
  //
  // `allow_zero` indicates to allow index i such that CDF[i] == CDF[i + 1].
  // This means that Pr(X = i) = 0, and alphabet i can neither be encoded nor
  // decoded. This is an error condition unless the user guarantees that
  // alphabet i never appears in data.
  //
  // NOTE: This function cannot detect whether the same CDF was used during
  // encoding and decoding. This function simply checks if CDF satisfies
  // required conditions.
  absl::Status CheckForError(absl::Span<const int16_t> cdf, int precision,
                             bool allow_zero = false) const {
    return CheckForErrorInternal(cdf, precision, allow_zero);
  }
  absl::Status CheckForError(absl::Span<const int32_t> cdf, int precision,
                             bool allow_zero = false) const {
    return CheckForErrorInternal(cdf, precision, allow_zero);
  }

 private:
  struct LinearSearch {
    // See the call site of F::Search() inside DecodeInternal().
    template <typename T>
    static const T* Search(uint64_t lower_bound, uint64_t size, const T* pv,
                           int64_t len) {
      return std::find_if(pv, pv + len, [lower_bound, size](T value) {
        return lower_bound <= size * static_cast<uint64>(value);
      });
    }
  };

  struct BinarySearch {
    // See the call site of F::Search() inside DecodeInternal().
    // REQUIRES: len > 0.
    template <typename T>
    static const T* Search(uint64_t lower_bound, uint64_t size, const T* pv,
                           int64_t len) {
      do {
        const auto half = len / 2;
        const T* mid = pv + half;
        if (lower_bound <= size * static_cast<uint64_t>(*mid)) {
          len = half;
        } else {
          pv = mid + 1;
          len -= half + 1;
        }
      } while (len > 0);
      return pv;
    }
  };

  template <typename F, typename T>
  int DecodeInternal(absl::Span<const T> cdf, int precision) {
    static_assert(sizeof(T) <= 4, "Type T is too large.");
    DCHECK_GT(precision, 0);
    DCHECK_LE(precision, 16);
    const uint64_t size = static_cast<uint64_t>(size_minus1_) + 1;
    const uint64_t lower_bound = (static_cast<uint64_t>(value_ - base_) + 1)
                                 << precision;

    // After the search, `pv` points to the smallest number v that satisfies
    // (value - base) < floor((size * v) / 2^precision), or that satisfies an
    // equivalent condition lower_bound <= size * v.
    //
    // It is a decode error if `pv` points to the first value of CDF. The
    // decoder assumes and does not check that cdf[0] == 0, i.e.,
    // size * cdf[0] = 0 <= (value - base) * 2^precision < lower_bound, and the
    // decoder start search from cdf[1].
    //
    // NOTE: We may also exclude cdf.back() if there is guarantee that the last
    // element of cdf is 2^precision, thus replacing cdf.size() - 1 with
    // cdf.size() - 2.
    DCHECK_GE(cdf.size(), 2);
    const T* pv = F::Search(lower_bound, size, cdf.data() + 1, cdf.size() - 1);

    // If (size * v) < lower_bound for all v in cdf, then pv points to one after
    // the last element of cdf. That is a decode error.
    // TODO(ssjhv): Consider returning -1 to indicate error. Or start len =
    // cdf.size() - 2 instead and give up detecting this error.
    DCHECK_LT(pv, cdf.data() + cdf.size());

    const uint32_t a = (size * static_cast<uint64_t>(*(pv - 1))) >> precision;
    const uint32_t b = ((size * static_cast<uint64_t>(*pv)) >> precision) - 1;
    DCHECK_LT(a, lower_bound >> precision);
    DCHECK_LE((lower_bound >> precision) - 1, b);

    base_ += a;
    size_minus1_ = b - a;

    if (size_minus1_ >> 16 == 0) {
      base_ <<= 16;
      size_minus1_ <<= 16;
      size_minus1_ |= 0xFFFF;

      Read16BitValue();
    }

    return static_cast<int>(pv - cdf.data() - 1);
  }

  void Read16BitValue() {
    value_ <<= 8;
    if (current_ != end_) {
      value_ |= static_cast<uint8_t>(*current_++);
    }
    value_ <<= 8;
    if (current_ != end_) {
      value_ |= static_cast<uint8_t>(*current_++);
    }
  }

  template <typename T>
  absl::Status CheckForErrorInternal(absl::Span<const T> cdf, int precision,
                                     bool allow_zero) const;

  uint32_t base_ = 0;
  uint32_t size_minus1_ = std::numeric_limits<uint32_t>::max();
  uint32_t value_ = 0;

  absl::string_view::const_iterator current_;
  absl::string_view::const_iterator end_;
};
}  // namespace tensorflow_compression

#endif  // TENSORFLOW_COMPRESSION_CC_LIB_RANGE_CODER_H_
