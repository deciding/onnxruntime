// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "matmul_bias.h"
#include "matmul_bias_helper.h"
#include "core/providers/cpu/math/gemm_matmul_common.h" // GemmPackBFp32
#include "core/util/math.h"

// #include "core/util/math_cpuonly.h"
#include "core/mlas/inc/mlas.h"

// #include "core/framework/tensorprotoutils.h"
// #include "onnx/defs/tensor_proto_util.h"
// #include "core/common/safeint.h"
// #include "core/framework/tensor.h"
// #include "core/platform/threadpool.h"
// #include "core/providers/common.h"
// #include "core/util/math_cpuonly.h"
// #include "core/mlas/inc/mlas.h"
// #include "core/framework/op_kernel.h"

namespace onnxruntime {
namespace contrib {


// register the opkernel to MS domain
ONNX_OPERATOR_TYPED_KERNEL_EX(
    MatMulBias,
    kMSDomain,
    1,
    float,
    kCpuExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    MatMulBias<float>);


// general: not supported, since it uses a math function version of Gemm and I don't want to change that file now
template <typename T>
Status MatMulBias<T>::Compute(OpKernelContext* ctx) const {
  concurrency::ThreadPool* thread_pool = ctx->GetOperatorThreadPool();

  const auto* a = ctx->Input<Tensor>(0);
  const auto* b = ctx->Input<Tensor>(1);
  const auto* bias = ctx->Input<Tensor>(2);

  MatMulBiasComputeHelper helper;
  ORT_RETURN_IF_ERROR(helper.Compute(a->Shape(), b->Shape(), bias->Shape()));
  Tensor* y = ctx->Output(0, helper.OutputShape());

  // Bail out early if the output is going to be empty
  if (y->Shape().Size() == 0)
    return Status::OK();

  // Using DataRaw as int32_t/uint32_t and int64_t/uint64_t share a common
  // operator body.
  const auto* a_data = reinterpret_cast<const T*>(a->DataRaw());
  const auto* b_data = reinterpret_cast<const T*>(b->DataRaw());
  const auto* bias_data = reinterpret_cast<const T*>(bias->DataRaw());
  auto* y_data = reinterpret_cast<T*>(y->MutableDataRaw());

  // TODO: replace it with GemmBatch for performance, it's OK for now as GemmBatch unrolls as well
  // do not support bias currently
  size_t max_len = helper.OutputOffsets().size();
  for (size_t i = 0; i < max_len; i++) {
    math::MatMul<T>(
        static_cast<int>(helper.M()),
        static_cast<int>(helper.N()),
        static_cast<int>(helper.K()),
        a_data + helper.LeftOffsets()[i],
        b_data + helper.RightOffsets()[i],
        y_data + helper.OutputOffsets()[i],
        thread_pool);
  }

  return Status::OK();
}


Status MatMulBias<float>::PrePack(const Tensor& tensor, int input_idx, bool& is_packed) {
  is_packed = false;

  // only pack Matrix B
  if (input_idx == 1) {
    is_packed = GemmPackBFp32KN(Info(), tensor, trans_b_attr_, packed_b_, b_shape_, 768);
  }
  return Status::OK();
}

Status MatMulBias<float>::Compute(OpKernelContext* ctx) const {
  concurrency::ThreadPool* thread_pool = ctx->GetOperatorThreadPool();

  const Tensor* a = ctx->Input<Tensor>(0);
  const Tensor* b = packed_b_ ? nullptr : ctx->Input<Tensor>(1);
  const auto& b_shape = b ? b->Shape() : b_shape_;
  const Tensor* bias = ctx->Input<Tensor>(2);

  // match CUDA kernel implementation, ignore transpose for vectors
  const bool trans_a = trans_a_attr_ && a->Shape().NumDimensions() != 1;
  const bool trans_b = trans_b_attr_ && b_shape.NumDimensions() != 1;

  MatMulBiasComputeHelper helper;
  ORT_RETURN_IF_ERROR(helper.Compute(a->Shape(), b_shape, bias->Shape(), trans_a, trans_b)); // no need bias shape for check for now
  Tensor* y = ctx->Output(0, helper.OutputShape());

  // Bail out early if the output is going to be empty
  if (y->Shape().Size() == 0)
    return Status::OK();

  const auto* a_data = a->Data<float>();
  const auto* b_data = b ? b->Data<float>() : nullptr;
  const auto* bias_data = bias->Data<float>();
  auto* y_data = y->MutableData<float>();

  const size_t max_len = helper.OutputOffsets().size();
  const size_t M = static_cast<size_t>(helper.M());
  const size_t N = static_cast<size_t>(helper.N());
  const size_t K = static_cast<size_t>(helper.K());
  const size_t lda = static_cast<int>(trans_a ? M : K);
  const size_t ldb = static_cast<int>(trans_b ? K : N);

  std::vector<MLAS_SGEMM_DATA_PARAMS> data(max_len);
  for (size_t i = 0; i < max_len; i++) {
    data[i].BIsPacked = bool(packed_b_);
    data[i].A = a_data + helper.LeftOffsets()[i];
    data[i].lda = lda;
    data[i].B = data[i].BIsPacked ? (float*)packed_b_.get() : b_data + helper.RightOffsets()[i];
    data[i].ldb = ldb;
    data[i].C = y_data + helper.OutputOffsets()[i];
    data[i].ldc = N;
    data[i].Bias = bias_data + helper.BiasOffsets()[i];
    data[i].alpha = alpha_attr_;
    data[i].beta = 0.0f;
  }
  // change function
  //MlasGemmBatch(trans_a ? CblasTrans : CblasNoTrans, trans_b ? CblasTrans : CblasNoTrans,
  //    M, N, K, data.data(), max_len, thread_pool);
  MlasGemmBatchKN(trans_a ? CblasTrans : CblasNoTrans, trans_b ? CblasTrans : CblasNoTrans,
      M, N, K, data.data(), max_len, thread_pool, 768, 128, concurrency::ThreadPool::DegreeOfParallelism(thread_pool), M>=N);

  //FILE *f = fopen("Ab.data", "wb");
  //fwrite(data[0].A, sizeof(float), M * K, f);
  //fclose(f);
  //f = fopen("Bb.data", "wb");
  //fwrite(data[0].B, sizeof(float), N * K, f);
  //fclose(f);
  //f = fopen("Cb.data", "wb");
  //fwrite(data[0].C, sizeof(float), M * N, f);
  //fclose(f);
  //f = fopen("bias.data", "wb");
  //fwrite(data[0].Bias, sizeof(float), N, f);
  //fclose(f);

  return Status::OK();
}

} // contrib
} // onnxruntime