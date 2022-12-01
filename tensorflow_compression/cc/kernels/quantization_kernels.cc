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
#include <cmath>
#include <cstdint>
#include <random>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/platform/status.h"

namespace tensorflow_compression {
namespace {
namespace errors = tensorflow::errors;
using tensorflow::DEVICE_CPU;
using tensorflow::OpKernel;
using tensorflow::OpKernelConstruction;
using tensorflow::OpKernelContext;
using tensorflow::Tensor;

// Xoroshiro256+ algorithm, adapted from
// https://prng.di.unimi.it/xoshiro256plus.c
inline uint64_t next_random(uint64_t* state) {
  const uint64_t result = state[0] + state[3];
  const uint64_t t = state[1] << 17;
  state[2] ^= state[0];
  state[3] ^= state[1];
  state[1] ^= state[2];
  state[0] ^= state[3];
  state[2] ^= t;
  state[3] = (state[3] << 45) | (state[3] >> (64 - 45));
  return result;
}

template <typename T>
class StochasticRoundOp : public OpKernel {
 public:
  explicit StochasticRoundOp(OpKernelConstruction* context)
      : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    const Tensor& inputs_tensor = context->input(0);
    auto inputs = inputs_tensor.flat<T>();

    OP_REQUIRES(context, context->input(1).dims() == 0,
                errors::InvalidArgument("step_size must be a scalar."));
    const float step_size = context->input(1).scalar<float>()();

    auto seed = context->input(2).flat<int32_t>();

    Tensor* outputs_tensor;
    OP_REQUIRES_OK(context, context->allocate_output(0, inputs_tensor.shape(),
                                                     &outputs_tensor));
    auto outputs = outputs_tensor->flat<int32_t>();

    uint64_t random_state[4];

    if (seed.size()) {
      std::seed_seq seq(seed.data(), seed.data() + seed.size());
      seq.generate(reinterpret_cast<uint32_t*>(random_state),
                   reinterpret_cast<uint32_t*>(random_state + 4));
    } else {
      // Seed the random state from system clock, in a best-effort fashion.
      uint64_t seed =
          std::chrono::high_resolution_clock::now().time_since_epoch().count();
      std::seed_seq seq{seed, seed >> 32};
      seq.generate(reinterpret_cast<uint32_t*>(random_state),
                   reinterpret_cast<uint32_t*>(random_state + 4));
    }

    for (int64_t i = 0; i < inputs.size(); ++i) {
      // Promote 16-bit types to 32 bit.
      float number = static_cast<float>(inputs(i)) / step_size;
      float integral = std::floor(number);
      outputs(i) = integral;
      // Regardless of T, comparing in float32 is accurate enough here.
      float fractional = number - integral;
      float random =
          (next_random(random_state) >> 40) * 0x1.0p-24f;  // from [0, 1)
      if (random < fractional) {
        ++outputs(i);
      }
    }
  }
};

REGISTER_KERNEL_BUILDER(Name("StochasticRound")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<tensorflow::bfloat16>("T"),
                        StochasticRoundOp<tensorflow::bfloat16>);
REGISTER_KERNEL_BUILDER(
    Name("StochasticRound").Device(DEVICE_CPU).TypeConstraint<Eigen::half>("T"),
    StochasticRoundOp<Eigen::half>);
REGISTER_KERNEL_BUILDER(
    Name("StochasticRound").Device(DEVICE_CPU).TypeConstraint<float>("T"),
    StochasticRoundOp<float>);

}  // namespace
}  // namespace tensorflow_compression
