// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/framework/op_kernel.h"

namespace onnxruntime {
namespace contrib {

template <typename T>
class MatMulBias final : public OpKernel {
 public:
  MatMulBias(const OpKernelInfo& info) : OpKernel(info) {}

  Status Compute(OpKernelContext* context) const override;
};

template <>
class MatMulBias<float> final : public OpKernel {
 public:
  MatMulBias(const OpKernelInfo& info) : OpKernel(info) {
    info.GetAttrOrDefault<int64_t>("transA", &trans_a_attr_, 0);
    info.GetAttrOrDefault<int64_t>("transB", &trans_b_attr_, 0);
    info.GetAttrOrDefault<float>("alpha", &alpha_attr_, 1.0);
  }

  Status PrePack(const Tensor& tensor, int input_idx, bool& is_packed) override;

  Status Compute(OpKernelContext* context) const override;

 private:
  TensorShape b_shape_;
  BufferUniquePtr packed_b_;

  // For FusedMatMul contrib ops
  float alpha_attr_;
  int64_t trans_a_attr_;
  int64_t trans_b_attr_;
};

}  // namespace contrib
}  // namespace onnxruntime
