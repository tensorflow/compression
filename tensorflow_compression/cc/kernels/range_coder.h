/* Copyright 2018 Google LLC. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPRESSION_CC_KERNELS_RANGE_CODER_H_
#define TENSORFLOW_COMPRESSION_CC_KERNELS_RANGE_CODER_H_

#include <limits>
#include <string>

#include "absl/types/span.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow_compression {

class RangeEncoder {
 public:
  RangeEncoder() = default;

  // Encodes a half-open interval [lower / 2^precision, upper / 2^precision).
  // Suppose each character to be encoded is from an integer-valued
  // distribution. When encoding a random character x0, the arguments lower and
  // upper represent
  //   Pr(X < x0) = lower / 2^precision,
  //   Pr(X < x0 + 1) = upper / 2^precision,
  // where X is a random variable following the distribution.
  //
  // For example, assume that the distribution has possible outputs 0, 1, 2, ...
  // To encode value 0, lower = 0 and upper = Pr(X = 0).
  // To encode value 1, lower = Pr(X = 0) and upper = Pr(X = 0 or 1).
  // To encode value 2, lower = Pr(X = 0 or 1) and upper = Pr(X = 0, 1, or 2).
  // ...
  //
  // REQUIRES: 0 <= lower < upper <= 2^precision.
  // REQUIRES: 0 < precision <= 16.
  void Encode(tensorflow::int32 lower, tensorflow::int32 upper, int precision,
              tensorflow::tstring* sink);

  // The encode may contain some under-determined values from previous encoding.
  // After Encode() calls, Finalize() must be called. Otherwise the encoded
  // string may not be decoded.
  void Finalize(tensorflow::tstring* sink);

 private:
  tensorflow::uint32 base_ = 0;
  tensorflow::uint32 size_minus1_ =
      std::numeric_limits<tensorflow::uint32>::max();
  tensorflow::uint64 delay_ = 0;
};

class RangeDecoder {
 public:
  // Holds a reference to `source`. The caller has to make sure that `source`
  // outlives the decoder object.
  explicit RangeDecoder(const tensorflow::tstring& source);

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
  // REQUIRES: cdf[cdf.size() - 1] <= 2^precision.
  // REQUIRES: 0 < precision <= 16.
  //
  // In practice the last element of `cdf` should equal to 2^precision.
  tensorflow::int32 Decode(absl::Span<const tensorflow::int32> cdf,
                           int precision);

 private:
  void Read16BitValue();

  tensorflow::uint32 base_ = 0;
  tensorflow::uint32 size_minus1_ =
      std::numeric_limits<tensorflow::uint32>::max();
  tensorflow::uint32 value_ = 0;

  tensorflow::tstring::const_iterator current_;
  const tensorflow::tstring::const_iterator end_;
};

}  // namespace tensorflow_compression

#endif  // TENSORFLOW_COMPRESSION_CC_KERNELS_RANGE_CODER_H_
