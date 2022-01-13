/* Copyright 2019 Google LLC. All Rights Reserved.

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

#include <algorithm>
#include <cmath>
#include <memory>
#include <random>
#include <vector>

#include "absl/types/span.h"
#include "tensorflow/core/common_runtime/shape_refiner.h"
#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/node_def.proto.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/types.proto.h"
#include "tensorflow/core/framework/versions.proto.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/graph/testlib.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/lib/core/bits.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/random/distribution_sampler.h"
#include "tensorflow/core/lib/random/simple_philox.h"
#include "tensorflow/core/platform/stacktrace_handler.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/public/session_options.h"

namespace tensorflow_compression {
namespace {
namespace random = tensorflow::random;
namespace test = tensorflow::test;
using tensorflow::DT_INT32;
using tensorflow::DT_STRING;
using tensorflow::Graph;
using tensorflow::int16;
using tensorflow::int32;
using tensorflow::int64;
using tensorflow::Node;
using tensorflow::NodeBuilder;
using tensorflow::NodeDefBuilder;
using tensorflow::OpRegistry;
using tensorflow::OpsTestBase;
using tensorflow::ShapeRefiner;
using tensorflow::Status;
using tensorflow::string;
using tensorflow::Tensor;
using tensorflow::TensorShape;
using tensorflow::TTypes;
using tensorflow::uint32;
using tensorflow::uint64;
using tensorflow::uint8;

void BuildDataAndCdf(random::SimplePhilox* gen, Tensor* data_tensor,
                     const Tensor& index_tensor, Tensor* cdf_tensor,
                     Tensor* cdf_size_tensor, Tensor* offset_tensor,
                     int precision, float over_estimate = 1.2f) {
  CHECK_GT(precision, 0);

  TTypes<int32>::Flat data = data_tensor->flat<int32>();
  TTypes<int32>::ConstFlat index = index_tensor.flat<int32>();
  CHECK_EQ(data.size(), index.size());

  TTypes<int32>::Matrix cdf = cdf_tensor->matrix<int32>();
  CHECK_GE(cdf.dimension(1), 2);

  TTypes<int32>::Vec cdf_size = cdf_size_tensor->vec<int32>();
  CHECK_EQ(cdf_size.size(), cdf.dimension(0));

  TTypes<int32>::Vec offset = offset_tensor->vec<int32>();
  CHECK_EQ(offset.size(), cdf.dimension(0));

  std::vector<std::unique_ptr<random::DistributionSampler>> sampler;
  sampler.reserve(cdf.dimension(0));

  {
    constexpr float kMinParam = 0.05f;
    constexpr float kMaxParam = 0.95f;

    std::vector<float> weights(2 * cdf.dimension(1));
    CHECK_LE(cdf.dimension(1), weights.size());
    for (int64 i = 0; i < cdf.dimension(0); ++i) {
      int32* slice = &cdf(i, 0);
      slice[0] = 0;

      // Random p in [kMinParam, kMaxParam).
      const float p = kMinParam + (kMaxParam - kMinParam) * gen->RandFloat();

      float mass = (1 - p) * over_estimate;
      for (int64 j = 0; j < weights.size(); ++j) {
        if (j < cdf.dimension(1) - 1) {
          const int32 inc =
              std::max<int32>(1, std::rint(std::ldexp(mass, precision)));
          slice[j + 1] = std::min(slice[j] + inc, 1 << precision);

          if (slice[j] < slice[j + 1]) {
            cdf_size(i) = j + 2;
          }
        }

        weights[j] = mass;
        mass *= p;
      }

      // Random offset between -32 and 31.
      offset(i) = gen->Uniform(64) - 32;

      // DistributionSampler class is not copyable. Therefore pointers to the
      // objects in the heap are stored.
      sampler.emplace_back(new random::DistributionSampler(weights));
    }
  }

  for (int64 i = 0; i < data.size(); ++i) {
    const int32 current_index = index(i);
    CHECK_GE(current_index, 0);
    CHECK_LT(current_index, sampler.size());

    data(i) = sampler[current_index]->Sample(gen);
  }
}

class UnboundedIndexRangeCoderOpsTest : public OpsTestBase {
 protected:
  Status RunEncodeOpDebug(int precision, int overflow_width,
                          absl::Span<const Tensor> input) {
    Tensor unused;
    return RunEncodeOpImpl(precision, overflow_width, 1, input, &unused);
  }

  Status RunEncodeOp(int precision, int overflow_width,
                     absl::Span<const Tensor> input, Tensor* output) {
    return RunEncodeOpImpl(precision, overflow_width, 0, input, output);
  }

  Status RunEncodeOpImpl(int precision, int overflow_width, int debug_level,
                         absl::Span<const Tensor> input, Tensor* output) {
    NodeDefBuilder builder("encode", "UnboundedIndexRangeEncode");
    for (const Tensor& tensor : input) {
      builder.Input(tensorflow::FakeInput(tensor.dtype()));
    }
    TF_RETURN_IF_ERROR(builder.Attr("precision", precision)
                           .Attr("overflow_width", overflow_width)
                           .Attr("debug_level", debug_level)
                           .Finalize(node_def()));
    TF_RETURN_IF_ERROR(InitOp());

    inputs_.clear();
    std::vector<Tensor> copies(input.size());
    for (int i = 0; i < input.size(); ++i) {
      copies[i] = input[i];
      inputs_.emplace_back(&copies[i]);
    }

    TF_RETURN_IF_ERROR(RunOpKernel());

    *output = *GetOutput(0);
    inputs_.clear();

    return Status::OK();
  }

  Status RunDecodeOpDebug(int precision, int overflow_width,
                          absl::Span<const Tensor> input) {
    Tensor unused;
    return RunDecodeOpImpl(precision, overflow_width, 1, input, &unused);
  }

  Status RunDecodeOp(int precision, int overflow_width,
                     absl::Span<const Tensor> input, Tensor* output) {
    return RunDecodeOpImpl(precision, overflow_width, 0, input, output);
  }

  Status RunDecodeOpImpl(int precision, int overflow_width, int debug_level,
                         absl::Span<const Tensor> input, Tensor* output) {
    NodeDefBuilder builder("decode", "UnboundedIndexRangeDecode");
    for (const Tensor& tensor : input) {
      builder.Input(tensorflow::FakeInput(tensor.dtype()));
    }
    TF_RETURN_IF_ERROR(builder.Attr("precision", precision)
                           .Attr("overflow_width", overflow_width)
                           .Attr("debug_level", debug_level)
                           .Finalize(node_def()));
    TF_RETURN_IF_ERROR(InitOp());

    inputs_.clear();
    std::vector<Tensor> copies(input.size());
    for (int i = 0; i < input.size(); ++i) {
      copies[i] = input[i];
      inputs_.emplace_back(&copies[i]);
    }

    TF_RETURN_IF_ERROR(RunOpKernel());

    *output = *GetOutput(0);
    inputs_.clear();

    return Status::OK();
  }

  void TestEncodeAndDecode(int precision, int overflow_width,
                           const Tensor& data, const Tensor& index,
                           const Tensor& cdf, const Tensor& cdf_size,
                           const Tensor& offset) {
    Tensor encoded;
    TF_ASSERT_OK(RunEncodeOp(precision, overflow_width,
                             {data, index, cdf, cdf_size, offset}, &encoded));

    Tensor decoded;
    TF_ASSERT_OK(RunDecodeOp(precision, overflow_width,
                             {encoded, index, cdf, cdf_size, offset},
                             &decoded));

    EXPECT_EQ(decoded.dtype(), data.dtype());
    EXPECT_EQ(decoded.shape(), data.shape());
    EXPECT_EQ(decoded.tensor_data(), data.tensor_data());
  }

  template <int N>
  Tensor CreateBroadcastingIndex(const TensorShape& shape,
                                 std::initializer_list<int> broadcasting_axes) {
    CHECK_EQ(N, shape.dims());

    std::vector<bool> broadcasting(shape.dims(), false);
    for (int32 axis : broadcasting_axes) {
      broadcasting[axis] = true;
    }

    TensorShape temp_shape;
    for (int axis = 0; axis < shape.dims(); ++axis) {
      temp_shape.AddDim(broadcasting[axis] ? 1 : shape.dim_size(axis));
    }

    Tensor temp(DT_INT32, temp_shape);
    auto temp_buffer = temp.flat<int32>();
    std::iota(temp_buffer.data(), temp_buffer.data() + temp_buffer.size(), 0);

    Eigen::array<Eigen::DenseIndex, N> broadcast;
    for (int axis = 0; axis < shape.dims(); ++axis) {
      broadcast[axis] = broadcasting[axis] ? shape.dim_size(axis) : 1;
    }

    Tensor index(DT_INT32, shape);
    index.tensor<int32, N>() = temp.tensor<int32, N>().broadcast(broadcast);
    return index;
  }
};

TEST_F(UnboundedIndexRangeCoderOpsTest, RandomIndex) {
  constexpr int kPrecision = 14;
  constexpr int kOverflowWidth = 3;
  constexpr int kCdfCount = 10;
  constexpr int kCdfWidth = 40;

  Tensor data(DT_INT32, {1, 32, 32, 16});
  Tensor index(DT_INT32, data.shape());

  std::random_device rd;
  random::PhiloxRandom philox(rd(), rd());
  random::SimplePhilox gen(&philox);

  auto flat = index.flat<int32>();
  for (int64 i = 0; i < flat.size(); ++i) {
    flat(i) = gen.Uniform(kCdfCount);
  }

  Tensor cdf(DT_INT32, {kCdfCount, kCdfWidth + 1});
  Tensor cdf_size(DT_INT32, {kCdfCount});
  Tensor offset(DT_INT32, {kCdfCount});
  BuildDataAndCdf(&gen, &data, index, &cdf, &cdf_size, &offset, kPrecision);

  // Insert some out-of-range values manually.
  auto data_flat = data.flat<int32>();
  data_flat(0) = -3;
  data_flat(data_flat.size() - 1) = kCdfWidth + 5;

  TestEncodeAndDecode(kPrecision, kOverflowWidth, data, index, cdf, cdf_size,
                      offset);
}

TEST_F(UnboundedIndexRangeCoderOpsTest, EncoderDebug) {
  Tensor data(DT_INT32, {});
  data.scalar<int32>()() = 0;

  Tensor index(DT_INT32, {});
  index.scalar<int32>()() = 0;

  Tensor cdf(DT_INT32, {1, 4});
  cdf.flat<int32>().setValues({0, 16, 18, 32});

  Tensor cdf_size(DT_INT32, {1});
  cdf_size.vec<int32>().setValues({4});

  Tensor offset(DT_INT32, {1});
  offset.vec<int32>().setValues({1});

  auto status = RunEncodeOpDebug(5, 2, {data, index, cdf, cdf_size, offset});
  EXPECT_TRUE(status.ok());

#define EXPECT_STATUS_SUBSTR(message)                                 \
  {                                                                   \
    auto status =                                                     \
        RunEncodeOpDebug(5, 2, {data, index, cdf, cdf_size, offset}); \
    EXPECT_FALSE(status.ok());                                        \
    EXPECT_NE(status.error_message().find((message)), string::npos)   \
        << status.error_message();                                    \
  }

  index.scalar<int32>()() = -1;
  EXPECT_STATUS_SUBSTR("'index' has a value not in");
  index.scalar<int32>()() = 0;

  cdf_size.vec<int32>().setValues({1});
  EXPECT_STATUS_SUBSTR("'cdf_size' has a value not in");
  cdf_size.vec<int32>().setValues({5});
  EXPECT_STATUS_SUBSTR("'cdf_size' has a value not in");
  cdf_size.vec<int32>().setValues({4});

  cdf.flat<int32>().setValues({1, 16, 18, 32});
  EXPECT_STATUS_SUBSTR("cdf[0]=");
  cdf.flat<int32>().setValues({0, 16, 18, 31});
  EXPECT_STATUS_SUBSTR("cdf[^1]=");
  cdf.flat<int32>().setValues({0, 18, 16, 32});
  EXPECT_STATUS_SUBSTR("monotonic");
  cdf.flat<int32>().setValues({0, 16, 18, 32});

  Tensor temp = cdf;
  cdf = Tensor(DT_INT32, {4});
  EXPECT_STATUS_SUBSTR("'cdf' should be 2-D");
  cdf = Tensor(DT_INT32, {1, 2});
  EXPECT_STATUS_SUBSTR("cdf.dim_size(1) >= 3");
  cdf = temp;

  temp = cdf_size;
  cdf_size = Tensor(DT_INT32, {1, 1});
  EXPECT_STATUS_SUBSTR("'cdf_size' should be 1-D");
  cdf_size = Tensor(DT_INT32, {2});
  EXPECT_STATUS_SUBSTR("should match the number of rows");
  cdf_size = temp;

  temp = offset;
  offset = Tensor(DT_INT32, {1, 1});
  EXPECT_STATUS_SUBSTR("'offset' should be 1-D");
  offset = Tensor(DT_INT32, {2});
  EXPECT_STATUS_SUBSTR("should match the number of rows");
  offset = temp;

#undef EXPECT_STATUS_SUBSTR
}

TEST_F(UnboundedIndexRangeCoderOpsTest, DecoderDebug) {
  Tensor encoded(DT_STRING, {});

  Tensor index(DT_INT32, {});
  index.scalar<int32>()() = 0;

  Tensor cdf(DT_INT32, {1, 4});
  cdf.flat<int32>().setValues({0, 16, 18, 32});

  Tensor cdf_size(DT_INT32, {1});
  cdf_size.vec<int32>().setValues({4});

  Tensor offset(DT_INT32, {1});
  offset.vec<int32>().setValues({1});

  auto status = RunDecodeOpDebug(5, 2, {encoded, index, cdf, cdf_size, offset});
  EXPECT_TRUE(status.ok());
}

TEST_F(UnboundedIndexRangeCoderOpsTest, Broadcast1Axis) {
  constexpr int kPrecision = 9;
  constexpr int kOverflowWidth = 4;
  constexpr int kDimensionSize = 1 << kPrecision;
  constexpr int kCdfWidth = 64;

  std::random_device rd;
  random::PhiloxRandom philox(rd(), rd());
  random::SimplePhilox gen(&philox);

  Tensor data{DT_INT32, {1, kDimensionSize, kDimensionSize}};
  Tensor cdf(DT_INT32, TensorShape{kDimensionSize, kCdfWidth + 1});
  Tensor cdf_size(DT_INT32, {kDimensionSize});
  Tensor offset(DT_INT32, {kDimensionSize});

  {
    // Axis 1.
    Tensor index = CreateBroadcastingIndex<3>(data.shape(), {1});

    BuildDataAndCdf(&gen, &data, index, &cdf, &cdf_size, &offset, kPrecision);
    TestEncodeAndDecode(kPrecision, kOverflowWidth, data, index, cdf, cdf_size,
                        offset);
  }

  {
    // Axis 2.
    Tensor index = CreateBroadcastingIndex<3>(data.shape(), {2});

    BuildDataAndCdf(&gen, &data, index, &cdf, &cdf_size, &offset, kPrecision);
    TestEncodeAndDecode(kPrecision, kOverflowWidth, data, index, cdf, cdf_size,
                        offset);
  }
}

TEST_F(UnboundedIndexRangeCoderOpsTest, Broadcast2Axes) {
  constexpr int kPrecision = 13;
  constexpr int kOverflowWidth = 2;
  constexpr int kDimensionSize1 = 1 << (kPrecision / 2);
  constexpr int kDimensionSize2 = 1 << (kPrecision - kPrecision / 2);
  constexpr int kCdfWidth = 64;

  Tensor data{DT_INT32, {2, kDimensionSize1, kDimensionSize2, 7}};
  const Tensor index = CreateBroadcastingIndex<4>(data.shape(), {1, 2});

  std::random_device rd;
  random::PhiloxRandom philox(rd(), rd());
  random::SimplePhilox gen(&philox);

  Tensor cdf{DT_INT32, TensorShape{14, kCdfWidth + 1}};
  Tensor cdf_size{DT_INT32, TensorShape{14}};
  Tensor offset{DT_INT32, TensorShape{14}};
  BuildDataAndCdf(&gen, &data, index, &cdf, &cdf_size, &offset, kPrecision);
  TestEncodeAndDecode(kPrecision, kOverflowWidth, data, index, cdf, cdf_size,
                      offset);
}

TEST_F(UnboundedIndexRangeCoderOpsTest, DecoderShapeFn) {
  Tensor encoded_tensor(DT_STRING, TensorShape{2});
  Tensor index_tensor(DT_INT32, TensorShape{4, 6, 8});
  Tensor cdf_tensor(DT_INT32, TensorShape{4 * 6 * 8, 2});
  Tensor cdf_size_tensor(DT_INT32, TensorShape{4 * 6 * 8});
  Tensor offset_tensor(DT_INT32, TensorShape{4 * 6 * 8});

  Graph g{OpRegistry::Global()};
  Node* encoded = test::graph::Constant(&g, encoded_tensor);
  Node* index = test::graph::Constant(&g, index_tensor);
  Node* cdf = test::graph::Constant(&g, cdf_tensor);
  Node* cdf_size = test::graph::Constant(&g, cdf_size_tensor);
  Node* offset = test::graph::Constant(&g, offset_tensor);
  Node* decode;
  TF_ASSERT_OK(
      NodeBuilder("range_decode", "UnboundedIndexRangeDecode", g.op_registry())
          .Input(encoded)
          .Input(index)
          .Input(cdf)
          .Input(cdf_size)
          .Input(offset)
          .Attr("precision", 10)
          .Attr("overflow_width", 3)
          .Finalize(&g, &decode));

  ShapeRefiner refiner{g.versions().producer(), g.op_registry()};
  TF_ASSERT_OK(refiner.AddNode(encoded));
  TF_ASSERT_OK(refiner.AddNode(index));
  TF_ASSERT_OK(refiner.AddNode(cdf));
  TF_ASSERT_OK(refiner.AddNode(cdf_size));
  TF_ASSERT_OK(refiner.AddNode(offset));
  TF_ASSERT_OK(refiner.AddNode(decode));

  auto* context = refiner.GetContext(decode);
  ASSERT_NE(context, nullptr);

  ASSERT_EQ(context->num_outputs(), 1);
  auto shape_handle = context->output(0);

  ASSERT_EQ(context->Rank(shape_handle), 3);
  EXPECT_EQ(context->Value(context->Dim(shape_handle, 0)), 4);
  EXPECT_EQ(context->Value(context->Dim(shape_handle, 1)), 6);
  EXPECT_EQ(context->Value(context->Dim(shape_handle, 2)), 8);
}

}  // namespace
}  // namespace tensorflow_compression

GTEST_API_ int main(int argc, char** argv) {
  tensorflow::testing::InstallStacktraceHandler();
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
