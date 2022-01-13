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

#include "tensorflow_compression/cc/lib/range_coder.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <vector>

#include "testing/base/public/gmock.h"
#include "testing/base/public/gunit.h"
#include "absl/random/random.h"
#include "absl/strings/string_view.h"
#include "util/task/status.h"

namespace tensorflow_compression {
namespace {
using absl::StatusCode;
using testing::HasSubstr;
using testing::status::IsOk;
using testing::status::StatusIs;

template <typename T>
void RangeEncodeDecodeTest(int precision, absl::BitGen* gen) {
  constexpr int kAlphabetSize = 10;

  const int32_t multiplier = (precision > 7) ? 32 : 1;
  std::vector<int32_t> histogram(kAlphabetSize, multiplier - 1);

  const int32_t data_size =
      (multiplier << precision) - histogram.size() * (multiplier - 1);
  CHECK_GE(data_size, 0);

  std::vector<uint8_t> data(data_size);
  static_assert(kAlphabetSize - 1 <= std::numeric_limits<uint8_t>::max(),
                "Alphabet size too large.");
  for (uint8_t& x : data) {
    x = absl::LogUniform(*gen, 0, kAlphabetSize - 1);
    ++histogram[x];
  }

  std::vector<T> cdf(histogram.size() + 1, 0);
  int32_t partial_sum = 0;
  for (int i = 0; i < histogram.size(); ++i) {
    partial_sum += histogram[i];
    cdf[i + 1] = partial_sum / multiplier;
  }

  ASSERT_EQ(cdf.front(), 0);
  ASSERT_EQ(cdf.back(), 1 << precision);

  std::vector<double> ideal_code_length(histogram.size());
  const double normalizer = static_cast<double>(1 << precision);
  for (int i = 0; i < ideal_code_length.size(); ++i) {
    ideal_code_length[i] = -std::log2((cdf[i + 1] - cdf[i]) / normalizer);
  }

  RangeEncoder encoder;
  std::string encoded;
  double ideal_length = 0.0;
  for (uint8_t x : data) {
    encoder.Encode(cdf[x], cdf[x + 1], precision, &encoded);
    ideal_length += ideal_code_length[x];
  }
  encoder.Finalize(&encoded);

  LOG(INFO) << "Encoded string length (bits): " << 8 * encoded.size()
            << ", whereas ideal " << ideal_length << " ("
            << (8 * encoded.size()) / ideal_length << " of ideal) "
            << " (ideal compression rate " << ideal_length / (8 * data.size())
            << ")";

  RangeDecoder decoder0(encoded);
  RangeDecoder decoder1(encoded);
  for (int i = 0; i < data.size(); ++i) {
    int32_t decoded = decoder0.Decode(cdf, precision);
    ASSERT_EQ(decoded, static_cast<int32_t>(data[i]));

    decoded = decoder1.DecodeLinearly(cdf, precision);
    ASSERT_EQ(decoded, static_cast<int32_t>(data[i]));
  }
  EXPECT_TRUE(decoder0.Finalize());
  EXPECT_TRUE(decoder1.Finalize());
}

TEST(RangeCoderTest, Precision1To11) {
  absl::BitGen gen;
  const int precision = 1 + absl::Uniform(gen, 0, 12);
  RangeEncodeDecodeTest<int16_t>(precision, &gen);
  RangeEncodeDecodeTest<int32_t>(precision, &gen);
}

TEST(RangeCoderTest, Precision12To16) {
  absl::BitGen gen;
  for (int precision = 12; precision < 15; ++precision) {
    RangeEncodeDecodeTest<int16_t>(precision, &gen);
    RangeEncodeDecodeTest<int32_t>(precision, &gen);
  }
  for (int precision = 15; precision < 17; ++precision) {
    RangeEncodeDecodeTest<int32_t>(precision, &gen);
  }
}

TEST(RangeCoderTest, FinalizeState0) {
  constexpr int kPrecision = 2;

  std::string output;
  RangeEncoder encoder;
  encoder.Encode(0, 2, kPrecision, &output);
  encoder.Finalize(&output);

  RangeDecoder decoder(output);
  EXPECT_EQ(decoder.Decode(std::initializer_list<int32_t>{0, 2, 4}, kPrecision),
            0);
  EXPECT_TRUE(decoder.Finalize());
}

// This test is designed to trigger the code path of state #1 in
// RangeEncoder::Finalize(). When changing RangeEncoder implementation or this
// test, make sure to check the coverage.
TEST(RangeCoderTest, FinalizeState1) {
  constexpr int kPrecision = 16;
  constexpr int32_t kLower = (1 << (kPrecision - 1)) - 1;
  constexpr int32_t kUpper = (1 << (kPrecision - 1)) + 1;

  std::string output;
  RangeEncoder encoder;
  encoder.Encode(kLower, kUpper, kPrecision, &output);
  encoder.Encode(kLower, kUpper, kPrecision, &output);
  encoder.Encode(kLower, kUpper, kPrecision, &output);
  encoder.Finalize(&output);

  std::array<int32_t, 4> cdf = {0, kLower, kUpper, 1 << kPrecision};

  RangeDecoder decoder(output);
  EXPECT_EQ(decoder.Decode(cdf, kPrecision), 1);
  EXPECT_EQ(decoder.Decode(cdf, kPrecision), 1);
  EXPECT_EQ(decoder.Decode(cdf, kPrecision), 1);
  EXPECT_TRUE(decoder.Finalize());
}

TEST(RangeCoderTest, Empty) {
  std::string output;
  RangeEncoder().Finalize(&output);
  EXPECT_THAT(output, testing::IsEmpty());

  RangeDecoder decoder(output);
  EXPECT_TRUE(decoder.Finalize());
}

TEST(RangeCoderTest, CheckReadToEnd) {
  std::string output;
  RangeEncoder().Finalize(&output);
  EXPECT_THAT(output, testing::IsEmpty());

  output.append(128, 0);
  ASSERT_THAT(output.size(), testing::Gt(4));

  RangeDecoder decoder(output);
  EXPECT_FALSE(decoder.Finalize());
}

TEST(RangeCoderTest, StateError) {
  constexpr int kPrecision = 10;
  constexpr std::array<int32_t, 3> kCdf = {0, 1, 1 << kPrecision};

  std::string output;
  RangeEncoder encoder;
  encoder.Encode(kCdf[1], kCdf[2], kPrecision, &output);
  encoder.Finalize(&output);

  ASSERT_THAT(output, testing::SizeIs(1));
  output.append(3, 0xFF);

  RangeDecoder decoder(output);
  EXPECT_EQ(decoder.Decode(kCdf, kPrecision), 1);
  EXPECT_FALSE(decoder.Finalize());
}

TEST(RangeCoderTest, EncoderCheckForError) {
  RangeEncoder encoder;
  EXPECT_THAT(encoder.CheckForError(0, 256, 0),
              StatusIs(StatusCode::kInvalidArgument, HasSubstr("precision")));
  EXPECT_THAT(encoder.CheckForError(0, 256, 17),
              StatusIs(StatusCode::kInvalidArgument, HasSubstr("precision")));
  EXPECT_THAT(
      encoder.CheckForError(256, 256, 10),
      StatusIs(StatusCode::kInvalidArgument, HasSubstr("lower < upper")));
  EXPECT_THAT(encoder.CheckForError(0, 256, 10), IsOk());
}

TEST(RangeCoderTest, DecoderCheckForError) {
  RangeDecoder decoder("");
  auto invalid_argument = [](const std::string& substr) {
    return StatusIs(StatusCode::kInvalidArgument, HasSubstr(substr));
  };

  // Execute code path for int16 cdf.
  {
    std::vector<int16_t> cdf = {0, 16, 18, 32};
    EXPECT_THAT(decoder.CheckForError(cdf, 0), invalid_argument("precision"));
    EXPECT_THAT(decoder.CheckForError(cdf, 17), invalid_argument("precision"));

    cdf = {};
    EXPECT_THAT(decoder.CheckForError(cdf, 5), invalid_argument("cdf.size"));
    cdf = {0};
    EXPECT_THAT(decoder.CheckForError(cdf, 5), invalid_argument("cdf.size"));
  }

  // Execute code path for int32 cdf.
  {
    std::vector<int32_t> cdf = {0, 16, 16, 32};
    EXPECT_THAT(decoder.CheckForError(cdf, 5, false),
                invalid_argument("monotonic"));
    EXPECT_THAT(decoder.CheckForError(cdf, 5, true), IsOk());

    cdf = {0, 17, 16, 32};
    EXPECT_THAT(decoder.CheckForError(cdf, 5, false),
                invalid_argument("monotonic"));
    EXPECT_THAT(decoder.CheckForError(cdf, 5, true),
                invalid_argument("monotonic"));

    cdf = {-1, 16, 18, 32};
    EXPECT_THAT(decoder.CheckForError(cdf, 5),
                invalid_argument("between 0 and 32"));

    cdf = {0, 16, 18, 33};
    EXPECT_THAT(decoder.CheckForError(cdf, 5),
                invalid_argument("between 0 and 32"));

    cdf = {0, 16, 18, 32};
    EXPECT_THAT(decoder.CheckForError(cdf, 5), IsOk());
  }
}

TEST(RangeCoderTest, DecoderCheckForStateDependentError) {
  std::string encoded;
  RangeEncoder encoder;
  encoder.Encode(16, 18, 5, &encoded);
  encoder.Finalize(&encoded);

  RangeDecoder decoder(encoded);

  std::vector<int32_t> cdf = {0, 16, 18, 32};
  EXPECT_THAT(decoder.CheckForError(cdf, 5), IsOk());

  cdf = {27, 28, 29, 30};
  EXPECT_THAT(
      decoder.CheckForError(cdf, 5),
      StatusIs(StatusCode::kInvalidArgument,
               HasSubstr("cdf[0]=27 is too large")));

  cdf = {0, 1, 2, 3};
  EXPECT_THAT(
      decoder.CheckForError(cdf, 5),
      StatusIs(StatusCode::kInvalidArgument,
               HasSubstr("cdf[^1]=3 is too small")));
}
}  // namespace
}  // namespace tensorflow_compression
