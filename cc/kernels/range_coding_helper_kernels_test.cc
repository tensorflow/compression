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

#include <algorithm>
#include <limits>
#include <random>

#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/node_def.proto.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/shape_inference_testutil.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/types.proto.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/lib/random/philox_random.h"
#include "tensorflow/core/lib/random/simple_philox.h"
#include "tensorflow/core/platform/stacktrace_handler.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow_compression {
namespace {
namespace gtl = tensorflow::gtl;
namespace random = tensorflow::random;
using tensorflow::DT_FLOAT;
using tensorflow::NodeDefBuilder;
using tensorflow::OpsTestBase;
using tensorflow::ShapeInferenceTestOp;
using tensorflow::Tensor;
using tensorflow::TensorShape;
using tensorflow::TTypes;

class PmfToQuantizedCdfOpTest : public OpsTestBase {
 protected:
  void SetupOp(int precision, Tensor* input) {
    TF_ASSERT_OK(NodeDefBuilder("pmf_to_cdf", "PmfToQuantizedCdf")
                     .Input(FakeInput(DT_FLOAT))
                     .Attr("precision", precision)
                     .Finalize(node_def()));
    TF_ASSERT_OK(InitOp());

    inputs_.clear();
    inputs_.emplace_back(input);
  }

  void GenerateData(random::SimplePhilox* rand,
                    gtl::MutableArraySlice<float> slice) {
    constexpr float minimum = std::numeric_limits<float>::epsilon();
    float sum = 0;
    for (float& value : slice) {
      value = std::max(rand->RandFloat(), minimum);
      sum += value;
    }
    for (float& value : slice) {
      value /= sum;
    }
  }

  void Verify(int precision, const Tensor& pmf_tensor,
              const Tensor& cdf_tensor) {
    ASSERT_EQ(pmf_tensor.dims(), cdf_tensor.dims());
    const int n = pmf_tensor.dims();

    for (int i = 0; i < n - 1; ++i) {
      EXPECT_EQ(pmf_tensor.dim_size(i), cdf_tensor.dim_size(i));
    }

    auto pmf = pmf_tensor.flat_inner_dims<float, 2>();
    auto cdf = cdf_tensor.flat_inner_dims<int32, 2>();
    EXPECT_EQ(pmf.dimension(1) + 1, cdf.dimension(1));

    const int normalizer = 1 << precision;
    for (int i = 0; i < pmf.dimension(0); ++i) {
      EXPECT_EQ(0, cdf(i, 0));

      TTypes<int32>::UnalignedConstVec cdf_slice(&cdf(i, 0), cdf.dimension(1));

      for (int j = 1; j < cdf_slice.size(); ++j) {
        const int32 diff = cdf_slice(j) - cdf_slice(j - 1);
        EXPECT_GT(diff, 0);
      }

      EXPECT_EQ(cdf_slice(cdf_slice.size() - 1), normalizer);
    }
  }
};

TEST_F(PmfToQuantizedCdfOpTest, UnderSum) {
  Tensor pmf(DT_FLOAT, {1, 10, 1, 32});
  auto matrix = pmf.flat_inner_dims<float, 2>();
  const std::size_t n = matrix.dimension(1);

  std::random_device rd;
  random::PhiloxRandom gen(rd(), rd());
  random::SimplePhilox rand(&gen);
  for (int64 i = 0; i < matrix.dimension(0); ++i) {
    GenerateData(&rand, {&matrix(i, 0), n});
  }

  pmf.flat<float>() = pmf.flat<float>() * 0.85f;

  constexpr int kPrecision = 10;
  SetupOp(kPrecision, &pmf);
  TF_ASSERT_OK(RunOpKernel());

  Verify(kPrecision, pmf, *GetOutput(0));
}

TEST_F(PmfToQuantizedCdfOpTest, OverSum) {
  Tensor pmf(DT_FLOAT, {10, 1, 1, 100});
  auto matrix = pmf.flat_inner_dims<float, 2>();

  // Half of each PMF is filled with zeros. The op will round up zeros to ones,
  // post quantization. These round ups are likely to make the sum over
  // normalizer value.
  matrix.setZero();
  const std::size_t n = matrix.dimension(1) / 2;

  std::random_device rd;
  random::PhiloxRandom gen(rd(), rd());
  random::SimplePhilox rand(&gen);
  for (int64 i = 0; i < matrix.dimension(0); ++i) {
    GenerateData(&rand, {&matrix(i, 0), n});
  }

  constexpr int kPrecision = 7;
  SetupOp(kPrecision, &pmf);
  TF_ASSERT_OK(RunOpKernel());

  Verify(kPrecision, pmf, *GetOutput(0));
}

TEST_F(PmfToQuantizedCdfOpTest, ShapeFn) {
  ShapeInferenceTestOp op("PmfToQuantizedCdf");

  INFER_OK(op, "?", "?");
  INFER_OK(op, "[3]", "[4]");
  INFER_OK(op, "[3,4]", "[d0_0,5]");
  INFER_OK(op, "[3,4,5]", "[d0_0,d0_1,6]");
}

class FingerprintOpTest : public tensorflow::OpsTestBase {
 protected:
  void MakeFingerprintOp(Tensor* tensor) {
    TF_ASSERT_OK(tensorflow::NodeDefBuilder("fingerprint", "ArrayFingerprint")
                     .Input(tensorflow::FakeInput(tensor->dtype()))
                     .Finalize(node_def()));
    TF_ASSERT_OK(InitOp());

    inputs_.clear();
    inputs_.emplace_back(tensor);
  }

  void MakeCheckFingerprintOp(Tensor* tensor, Tensor* fingerprint) {
    TF_ASSERT_OK(
        tensorflow::NodeDefBuilder("check_fingerprint", "CheckArrayFingerprint")
            .Input(tensorflow::FakeInput(tensor->dtype()))
            .Input(tensorflow::FakeInput(fingerprint->dtype()))
            .Finalize(node_def()));
    TF_ASSERT_OK(InitOp());

    inputs_.clear();
    inputs_.emplace_back(tensor);
    inputs_.emplace_back(fingerprint);
  }
};

TEST_F(FingerprintOpTest, Verify) {
  std::random_device rd;
  random::PhiloxRandom gen(rd(), rd());
  random::SimplePhilox rand(&gen);
  for (tensorflow::DataType dtype : tensorflow::kRealNumberTypes) {
    const int rank = rand.Uniform(4);

    TensorShape shape;
    for (int i = 0; i < rank; ++i) {
      shape.AddDim(rand.Uniform(9) + 1);
    }

    Tensor tensor(dtype, shape);

    const int64 length = shape.num_elements() * tensorflow::DataTypeSize(dtype);
    auto buffer = tensor.bit_casted_shaped<char, 1>({length});
    buffer.setRandom();

    MakeFingerprintOp(&tensor);
    TF_ASSERT_OK(RunOpKernel());

    Tensor fingerprint = *GetOutput(0);

    MakeCheckFingerprintOp(&tensor, &fingerprint);
    TF_ASSERT_OK(RunOpKernel());

    // Change one byte in the buffer.
    const int64 pos = rand.Uniform(length);
    buffer(pos) = ~buffer(pos);

    MakeCheckFingerprintOp(&tensor, &fingerprint);
    ASSERT_FALSE(RunOpKernel().ok());
  }
}

TEST_F(FingerprintOpTest, FingerprintShapeFn) {
  tensorflow::ShapeInferenceTestOp op("ArrayFingerprint");

  INFER_OK(op, "?", "[]");
  INFER_OK(op, "[]", "[]");
  INFER_OK(op, "[1]", "[]");
  INFER_OK(op, "[1,2]", "[]");
  INFER_OK(op, "[1,2,3]", "[]");
}

TEST_F(FingerprintOpTest, CheckFingerprintShapeFn) {
  tensorflow::ShapeInferenceTestOp op("CheckArrayFingerprint");

  INFER_OK(op, "?;?", "in0");
  INFER_OK(op, "[];?", "in0");
  INFER_OK(op, "[1,2];?", "in0");
  INFER_OK(op, "[1,2,3];?", "in0");
  INFER_ERROR("rank 0", op, "?;[1]");
}

}  // namespace
}  // namespace tensorflow_compression

GTEST_API_ int main(int argc, char** argv) {
  tensorflow::testing::InstallStacktraceHandler();
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
