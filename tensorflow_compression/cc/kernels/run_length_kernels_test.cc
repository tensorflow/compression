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
#include <bitset>
#include <cstdint>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "absl/types/span.h"
#include "tensorflow/core/common_runtime/shape_refiner.h"
#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/node_def.proto.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/types.proto.h"
#include "tensorflow/core/framework/versions.proto.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/graph/testlib.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/platform/stacktrace_handler.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow_compression/cc/lib/bit_coder.h"

namespace tensorflow_compression {
namespace {
namespace test = tensorflow::test;
using tensorflow::DT_INT32;
using tensorflow::DT_STRING;
using tensorflow::Graph;
using tensorflow::Node;
using tensorflow::NodeBuilder;
using tensorflow::NodeDefBuilder;
using tensorflow::OpRegistry;
using tensorflow::OpsTestBase;
using tensorflow::ShapeRefiner;
using tensorflow::Status;
using tensorflow::Tensor;
using tensorflow::TensorShape;
using tensorflow::tstring;

class BitCodingOpsTest : public OpsTestBase {
 protected:
  Status RunEncodeOp(absl::Span<const Tensor> inputs, Tensor* output,
                     const int run_length_code, const int magnitude_code,
                     const bool use_run_length_for_non_zeros) {
    TF_RETURN_IF_ERROR(
        NodeDefBuilder("encode", "RunLengthEncode")
            .Attr("run_length_code", run_length_code)
            .Attr("magnitude_code", magnitude_code)
            .Attr("use_run_length_for_non_zeros", use_run_length_for_non_zeros)
            .Input(tensorflow::FakeInput(DT_INT32))
            .Finalize(node_def()));
    TF_RETURN_IF_ERROR(InitOp());
    inputs_.clear();
    std::vector<Tensor> copies(inputs.size());
    for (int i = 0; i < inputs.size(); ++i) {
      copies[i] = inputs[i];
      inputs_.emplace_back(&copies[i]);
    }
    TF_RETURN_IF_ERROR(RunOpKernel());
    *output = *GetOutput(0);
    inputs_.clear();
    return tensorflow::OkStatus();
  }

  Status RunDecodeOp(absl::Span<const Tensor> inputs, Tensor* output,
                     const int run_length_code, const int magnitude_code,
                     const bool use_run_length_for_non_zeros) {
    TF_RETURN_IF_ERROR(
        NodeDefBuilder("decode", "RunLengthDecode")
            .Attr("run_length_code", run_length_code)
            .Attr("magnitude_code", magnitude_code)
            .Attr("use_run_length_for_non_zeros", use_run_length_for_non_zeros)
            .Input(tensorflow::FakeInput(DT_STRING))
            .Input(tensorflow::FakeInput(DT_INT32))
            .Finalize(node_def()));
    TF_RETURN_IF_ERROR(InitOp());
    inputs_.clear();
    std::vector<Tensor> copies(inputs.size());
    for (int i = 0; i < inputs.size(); ++i) {
      copies[i] = inputs[i];
      inputs_.emplace_back(&copies[i]);
    }
    TF_RETURN_IF_ERROR(RunOpKernel());
    *output = *GetOutput(0);
    inputs_.clear();
    return tensorflow::OkStatus();
  }

  void TestEncodeAndDecode(const Tensor& data_tensor, const int run_length_code,
                           const int magnitude_code,
                           const bool use_run_length_for_non_zeros) {
    Tensor code_tensor;
    TF_ASSERT_OK(RunEncodeOp({data_tensor}, &code_tensor, run_length_code,
                             magnitude_code, use_run_length_for_non_zeros));
    const TensorShape& data_shape = data_tensor.shape();
    Tensor shape_tensor{DT_INT32, {data_shape.dims()}};
    for (int i = 0; i < data_shape.dims(); ++i) {
      shape_tensor.flat<int32_t>()(i) = data_shape.dim_size(i);
    }
    Tensor decoded_tensor;
    TF_ASSERT_OK(RunDecodeOp({code_tensor, shape_tensor}, &decoded_tensor,
                             run_length_code, magnitude_code,
                             use_run_length_for_non_zeros));
    EXPECT_EQ(decoded_tensor.dtype(), data_tensor.dtype());
    EXPECT_EQ(decoded_tensor.shape(), data_tensor.shape());
    EXPECT_EQ(decoded_tensor.tensor_data(), data_tensor.tensor_data());
  }
};

TEST_F(BitCodingOpsTest, EncodeAndDecode) {
  Tensor data_tensor{DT_INT32, {5, 70, 2}};
  auto data = data_tensor.flat<int32_t>();
  for (int i = 0; i < data.size(); ++i) {
    data.data()[i] = i % 2 ? i : -i;
  }
  TestEncodeAndDecode(data_tensor, 2, -1, false);
  TestEncodeAndDecode(data_tensor, -1, 5, true);
  TestEncodeAndDecode(data_tensor, -1, 4, false);
  TestEncodeAndDecode(data_tensor, 3, -1, true);
}

TEST_F(BitCodingOpsTest, EncodeAndDecodeLeadingZeros) {
  Tensor data_tensor{DT_INT32, {2, 80, 2}};
  auto data = data_tensor.flat<int32_t>();
  for (int i = 0; i < data.size(); ++i) {
    if (i < data.size() / 2) {
      data.data()[i] = 0;
    } else {
      data.data()[i] = i % 2 ? i : -i;
    }
  }
  TestEncodeAndDecode(data_tensor, 2, -1, false);
  TestEncodeAndDecode(data_tensor, -1, 5, true);
  TestEncodeAndDecode(data_tensor, -1, 4, false);
  TestEncodeAndDecode(data_tensor, 3, -1, true);
}

TEST_F(BitCodingOpsTest, EncodeAndDecodeTrailingZeros) {
  Tensor data_tensor{DT_INT32, {50, 7, 2}};
  auto data = data_tensor.flat<int32_t>();
  for (int i = 0; i < data.size(); ++i) {
    if (i > data.size() / 2) {
      data.data()[i] = 0;
    } else {
      data.data()[i] = i % 2 ? i : -i;
    }
  }
  TestEncodeAndDecode(data_tensor, 2, -1, false);
  TestEncodeAndDecode(data_tensor, -1, 5, true);
  TestEncodeAndDecode(data_tensor, -1, 4, false);
  TestEncodeAndDecode(data_tensor, 3, -1, true);
}

TEST_F(BitCodingOpsTest, EncodeAndDecodeInterspersedZeros) {
  Tensor data_tensor{DT_INT32, {3, 7, 20}};
  auto data = data_tensor.flat<int32_t>();
  for (int i = 0; i < data.size(); ++i) {
    data.data()[i] = i % 2 ? i : 0;
  }
  TestEncodeAndDecode(data_tensor, 2, -1, false);
  TestEncodeAndDecode(data_tensor, -1, 5, true);
  TestEncodeAndDecode(data_tensor, -1, 4, false);
  TestEncodeAndDecode(data_tensor, 3, -1, true);
}

TEST_F(BitCodingOpsTest, DecoderShapeFn) {
  Tensor code_tensor{DT_STRING, {}};
  Tensor shape_tensor{DT_INT32, {3}};

  shape_tensor.flat<int32_t>().setValues({3, 2, 5});

  Graph g{OpRegistry::Global()};
  Node* code = test::graph::Constant(&g, code_tensor);
  Node* shape = test::graph::Constant(&g, shape_tensor);
  Node* decode;
  TF_ASSERT_OK(NodeBuilder("decode", "RunLengthDecode", g.op_registry())
               .Attr("run_length_code", -1)
               .Attr("magnitude_code", -1)
               .Attr("use_run_length_for_non_zeros", false)
               .Input(code)
               .Input(shape)
               .Finalize(&g, &decode));

  ShapeRefiner refiner{g.versions().producer(), g.op_registry()};
  TF_ASSERT_OK(refiner.AddNode(code));
  TF_ASSERT_OK(refiner.AddNode(shape));
  TF_ASSERT_OK(refiner.AddNode(decode));

  auto* context = refiner.GetContext(decode);
  ASSERT_NE(context, nullptr);

  ASSERT_EQ(context->num_outputs(), 1);
  auto shape_handle = context->output(0);

  ASSERT_EQ(context->Rank(shape_handle), 3);
  EXPECT_EQ(context->Value(context->Dim(shape_handle, 0)), 3);
  EXPECT_EQ(context->Value(context->Dim(shape_handle, 1)), 2);
  EXPECT_EQ(context->Value(context->Dim(shape_handle, 2)), 5);
}

TEST_F(BitCodingOpsTest, ManualEncodeWithBitcodingLibrary) {
  Tensor data_tensor(DT_INT32, {3});
  data_tensor.flat<int32_t>().setValues({0, -3, 1});

  Tensor code_tensor;
  TF_ASSERT_OK(RunEncodeOp({data_tensor}, &code_tensor, -1, -1, false));

  // Use bitcoding library to encode data.
  BitWriter enc_;
  enc_.WriteGamma(2);   // one zero
  enc_.WriteOneBit(0);  // negative
  enc_.WriteGamma(3);   // 3
  enc_.WriteGamma(1);   // no zeros
  enc_.WriteOneBit(1);  // positive
  enc_.WriteGamma(1);   // 1
  Tensor expected_code_tensor(DT_STRING, {});
  auto encoded = enc_.GetData();
  expected_code_tensor.scalar<tstring>()().assign(encoded.data(),
                                                  encoded.size());

  // Check that code_tensor has expected value.
  test::ExpectTensorEqual<tstring>(code_tensor, expected_code_tensor);
}

TEST_F(BitCodingOpsTest, ManualDecodeWithBitcodingLibrary) {
  // Use bitcoding library to manually encode [-3, 1, 0, 0] into code.
  BitWriter enc_;
  enc_.WriteGamma(1);   // no zeros
  enc_.WriteOneBit(0);  // negative
  enc_.WriteGamma(3);   // 3
  enc_.WriteGamma(1);   // no zeros
  enc_.WriteOneBit(1);  // positive
  enc_.WriteGamma(1);   // 1
  enc_.WriteGamma(3);   // two zeros
  Tensor code_tensor(DT_STRING, {});
  auto encoded = enc_.GetData();
  code_tensor.scalar<tstring>()().assign(encoded.data(), encoded.size());

  Tensor shape_tensor(DT_INT32, {1});
  shape_tensor.flat<int32_t>().setValues({4});

  Tensor data_tensor;
  TF_ASSERT_OK(
      RunDecodeOp({code_tensor, shape_tensor}, &data_tensor, -1, -1, false));

  Tensor expected_data_tensor(DT_INT32, {4});
  expected_data_tensor.flat<int32_t>().setValues({-3, 1, 0, 0});

  // Check that decoded data has expected values.
  test::ExpectTensorEqual<int32_t>(data_tensor, expected_data_tensor);
}

TEST_F(BitCodingOpsTest, EncodeConsistent) {
  Tensor data_tensor(DT_INT32, {4});
  data_tensor.flat<int32_t>().setValues({-6, 3, 0, 0});

  Tensor code_tensor;
  TF_ASSERT_OK(RunEncodeOp({data_tensor}, &code_tensor, -1, -1, false));

  char expected_code[] = {0b11010001, 0b01101101};

  Tensor expected_code_tensor(DT_STRING, {});
  expected_code_tensor.scalar<tstring>()().assign(expected_code, 2);

  test::ExpectTensorEqual<tstring>(code_tensor, expected_code_tensor);
}

TEST_F(BitCodingOpsTest, DecodeConsistent) {
  char code[] = {0b11010001, 0b01101101};  // [-6, 3, 0, 0]

  Tensor code_tensor(DT_STRING, {});
  code_tensor.scalar<tstring>()().assign(code, 2);

  Tensor shape_tensor(DT_INT32, {1});
  shape_tensor.flat<int32_t>().setValues({4});

  Tensor data_tensor;
  TF_ASSERT_OK(
      RunDecodeOp({code_tensor, shape_tensor}, &data_tensor, -1, -1, false));

  Tensor expected_data_tensor(DT_INT32, {4});
  expected_data_tensor.flat<int32_t>().setValues({-6, 3, 0, 0});

  // Check that decoded data has expected values.
  test::ExpectTensorEqual<int32_t>(data_tensor, expected_data_tensor);
}

// TODO(nicolemitchell,jonycgn) Add more corner cases to unit tests.
// Examples: decode empty string (null pointer), decode strings that end
// prematurely, decode long string of zeros that causes overflow in ReadGamma,
// decode incorrect run length that exceeds tensor size, encode int32::min
// tensor, encode tensor with very large values to ensure it doesn't exceed
// allocated buffer, encode gamma values <= 0, ...

}  // namespace
}  // namespace tensorflow_compression

GTEST_API_ int main(int argc, char** argv) {
  tensorflow::testing::InstallStacktraceHandler();
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
