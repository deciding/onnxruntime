// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "attention_cpu_base.h"
#include "attention_helper.h"
#include "core/framework/tensorprotoutils.h"
#include "core/graph/onnx_protobuf.h"
#include "core/util/math.h"
#include "core/util/math_cpuonly.h"
#include "core/common/safeint.h"
#include "core/platform/threadpool.h"

using onnxruntime::concurrency::ThreadPool;

namespace onnxruntime {
namespace contrib {

template <typename T>
class Attention : public OpKernel, public AttentionCPUBase {
 public:
  explicit Attention(const OpKernelInfo& info);

  Status Compute(OpKernelContext* context) const override;
  Status PrePack(const Tensor& tensor, int input_idx, bool& is_packed) override;

 private:
  BufferUniquePtr packed_weights_;
  size_t packed_weights_size_ = 0;
  TensorShape weight_shape_;
};

// These ops are internal-only, so register outside of onnx
ONNX_OPERATOR_TYPED_KERNEL_EX(
    Attention,
    kMSDomain,
    1,
    float,
    kCpuExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    Attention<float>);

AttentionBase::AttentionBase(const OpKernelInfo& info) {
  int64_t num_heads = 0;
  ORT_ENFORCE(info.GetAttr("num_heads", &num_heads).IsOK() && num_heads > 0);
  num_heads_ = static_cast<int>(num_heads);

  is_unidirectional_ = info.GetAttrOrDefault<int64_t>("unidirectional", 0) == 1;
}

Status AttentionBase::CheckInputs(const TensorShape& input_shape,
                                  const TensorShape& weights_shape,
                                  const TensorShape& bias_shape,
                                  const Tensor*& mask_index,
                                  const Tensor* past) const {
  // Input shapes:
  //   input       : (batch_size, sequence_length, input_hidden_size)
  //   weights     : (input_hidden_size, 3 * hidden_size)
  //   bias        : (3 * hidden_size)
  //   mask_index  : nullptr, (batch_size), (2 * batch_size),
  //                 or (batch_size, 1), (1, 1)
  //                 or (batch_size, past_sequence_length + sequence_length)
  //                 or (batch_size, sequence_length, past_sequence_length + sequence_length)
  //   past        : (2, batch_size, num_heads, past_sequence_length, head_size)
  //
  // Where hidden_size = num_heads * head_size.
  // When a model is pruned (like some attention heads are removed), hidden_size < input_hidden_size.

  const auto& dims = input_shape.GetDims();
  if (dims.size() != 3) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Input 'input' is expected to have 3 dimensions, got ",
                           dims.size());
  }
  int batch_size = static_cast<int>(dims[0]);
  int sequence_length = static_cast<int>(dims[1]);

  const auto& weights_dims = weights_shape.GetDims();
  if (weights_dims.size() != 2) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Input 'weights' is expected to have 2 dimensions, got ",
                           weights_dims.size());
  }
  if (weights_dims[0] != dims[2]) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Input 1 dimension 0 should have same length as dimension 2 of input 0");
  }

  int hidden_size = static_cast<int>(weights_dims[1]) / 3;
  if (3 * hidden_size != static_cast<int>(weights_dims[1])) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Input 1 dimension 1 should be 3 times of hidden dimension");
  }

  if (hidden_size % num_heads_ != 0) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "hidden_size should be divisiable by num_heads.");
  }

  const auto& bias_dims = bias_shape.GetDims();
  if (bias_dims.size() != 1) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Input 'bias' is expected to have 1 dimension, got ",
                           bias_dims.size());
  }
  if (bias_dims[0] != weights_dims[1]) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Input 'bias' dimension 0 should have same length as dimension 1 of input 'weights'");
  }

  int past_sequence_length = 0;
  if (past != nullptr) {  // past is optional
    const auto& past_dims = past->Shape().GetDims();
    if (past_dims.size() != 5) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Input 'past' is expected to have 5 dimension, got ",
                             past_dims.size());
    }
    if (static_cast<int>(past_dims[0]) != 2) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Inputs 'past' dimension 0 shall have length of 2");
    }
    if (static_cast<int>(past_dims[1]) != batch_size) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Inputs 'past' dimension 1 shall have same length as dimension 0 of input 0");
    }
    if (static_cast<int>(past_dims[2]) != num_heads_) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Inputs 'past' dimension 2 shall have length of num_heads", num_heads_);
    }
    if (static_cast<int>(past_dims[4]) != hidden_size / num_heads_) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Inputs 'past' dimension 2 shall have length of ", hidden_size / num_heads_);
    }
    past_sequence_length = static_cast<int>(past_dims[3]);
  }

  if (mask_index != nullptr) {  // mask_index is optional
    const auto& mask_dims = mask_index->Shape().GetDims();
    if (mask_dims.size() == 1) {
      if (static_cast<int>(mask_dims[0]) != batch_size && static_cast<int>(mask_dims[0]) != 2 * batch_size) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Inputs 'mask_index' with 1D data shall have length of batch_size or 2 * batch_size");
      }
    } else if (mask_dims.size() == 2) {
      if (static_cast<int>(mask_dims[0]) != batch_size || static_cast<int>(mask_dims[1]) != past_sequence_length + sequence_length) {
        // Add operator supports broadcasting. Here we handle a case with only one element in the 2nd dimension.
        if ((static_cast<int>(mask_dims[0]) == batch_size || static_cast<int>(mask_dims[0]) == 1) && static_cast<int>(mask_dims[1]) == 1) {
          // Mask will have same value after propogation, which has same effect as no mask.
          mask_index = nullptr;
        } else {
          return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Inputs 'mask_index' with 2D data shall have shape batch_size x (past_sequence_length + sequence_length)");
        }
      }
    } else if (mask_dims.size() == 3) {
      if (static_cast<int>(mask_dims[0]) != batch_size || mask_dims[1] != sequence_length || static_cast<int>(mask_dims[2]) != past_sequence_length + sequence_length) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Inputs 'mask_index' with 3D data shall have shape batch_size x sequence_length x (past_sequence_length + sequence_length)");
      }
    } else {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Input 'mask_index' is expected to have 1, 2 or 3 dimensions, got ",
                             mask_dims.size());
    }
  }
  return Status::OK();
}

Status AttentionBase::CheckInputs(const TensorShape& input_shape,
                                  const TensorShape& weights_shape,
                                  const TensorShape& bias_shape,
                                  const Tensor*& mask_index,
                                  const Tensor* past,
                                  const int max_threads_per_block) const {
  if (num_heads_ > max_threads_per_block) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "num_heads should be no larger than ", max_threads_per_block);
  }

  return CheckInputs(input_shape, weights_shape, bias_shape, mask_index, past);
}

Tensor* AttentionBase::GetPresent(OpKernelContext* context,
                                  const Tensor* past,
                                  int batch_size,
                                  int head_size,
                                  int sequence_length,
                                  int& past_sequence_length) const {
  // Input and output shapes:
  //   past        : (2, batch_size, num_heads, past_sequence_length, head_size)
  //   present     : (2, batch_size, num_heads, past_sequence_length + sequence_length, head_size)

  std::vector<int64_t> present_dims{2, batch_size, num_heads_, sequence_length, head_size};
  if (nullptr != past) {
    const auto& past_dims = past->Shape().GetDims();
    past_sequence_length = static_cast<int>(past_dims[3]);
    present_dims[3] += past_dims[3];
  }

  TensorShape present_shape(present_dims);
  Tensor* present = context->Output(1, present_shape);
  if (nullptr != past && nullptr == present) {
    ORT_THROW("Expect to have present state output when past state input is given");
  }

  return present;
}

template <typename T>
Attention<T>::Attention(const OpKernelInfo& info) : OpKernel(info), AttentionCPUBase(info) {
}

template <typename T>
Status Attention<T>::PrePack(const Tensor& weights, int input_idx, bool& is_packed) {
  is_packed = false;

  if (1 != input_idx) {
    return Status::OK();
  }

  weight_shape_ = weights.Shape();
  const auto& weights_dims = weight_shape_.GetDims();
  if (weights_dims.size() != 2) {
    return Status::OK();
  }

  const size_t input_hidden_size = static_cast<size_t>(weights_dims[0]);
  const size_t hidden_size_x3 = static_cast<size_t>(weights_dims[1]);
  const size_t hidden_size = hidden_size_x3 / 3;
  const size_t head_size = hidden_size / num_heads_;

  // Bail out if the weights shape has an expected shape.
  if ((hidden_size == 0) || ((hidden_size % num_heads_) != 0) || (hidden_size_x3 != 3 * hidden_size)) {
    return Status::OK();
  }

  const auto* weights_data = weights.Data<T>();

  packed_weights_size_ = MlasGemmPackBSize(head_size, input_hidden_size);
  if (packed_weights_size_ == 0) {
    return Status::OK();
  }

  const size_t loop_len = static_cast<size_t>(3) * num_heads_;
  auto alloc = Info().GetAllocator(0, OrtMemTypeDefault);
  auto* packed_weights_data = static_cast<uint8_t*>(alloc->AllocArray(packed_weights_size_, loop_len));
  packed_weights_ = BufferUniquePtr(packed_weights_data, BufferDeleter(alloc));

  for (size_t i = 0; i < loop_len; i++) {
    //MlasGemmPackB(CblasNoTrans, head_size, input_hidden_size, weights_data, hidden_size_x3, packed_weights_data);
    // CHANGE1: StrideK
    // CHANGE6: QKV bias
    MlasGemmPackBKN(CblasNoTrans, head_size, input_hidden_size, weights_data, hidden_size_x3, packed_weights_data, 768);
    packed_weights_data += packed_weights_size_;
    weights_data += head_size;
  }

  is_packed = true;
  return Status::OK();
}

template <typename T>
Status Attention<T>::Compute(OpKernelContext* context) const {
  const Tensor* input = context->Input<Tensor>(0);
  const Tensor* weights = packed_weights_ ? nullptr : context->Input<Tensor>(1);
  const Tensor* bias = context->Input<Tensor>(2);
  const Tensor* mask_index = context->Input<Tensor>(3);
  const Tensor* past = context->Input<Tensor>(4);

  const TensorShape& weights_shape = (weights ? weights->Shape() : weight_shape_);
  ORT_RETURN_IF_ERROR(CheckInputs(input->Shape(),
                                  weights_shape,
                                  bias->Shape(),
                                  mask_index,
                                  past));

  const auto& shape = input->Shape().GetDims();
  const int batch_size = static_cast<int>(shape[0]);
  const int sequence_length = static_cast<int>(shape[1]);
  const int input_hidden_size = static_cast<int>(shape[2]);

  const auto& weights_dims = weights_shape.GetDims();
  const int hidden_size = static_cast<int>(weights_dims[1]) / 3;
  const int head_size = hidden_size / num_heads_;

  std::vector<int64_t> output_shape(3);
  output_shape[0] = shape[0];
  output_shape[1] = shape[1];
  output_shape[2] = static_cast<int64_t>(hidden_size);
  Tensor* output = context->Output(0, output_shape);

  constexpr size_t element_size = sizeof(T);

  AllocatorPtr allocator;
  ORT_RETURN_IF_ERROR(context->GetTempSpaceAllocator(&allocator));

  auto* tp = context->GetOperatorThreadPool();
  // Compute Q, K, V
  // gemm_data(BS, 3NH) = input(BS, D) x weights(D, 3NH) + bias(3NH)
  // D (input_hidden_size) is hidden dimension of input, where D could be larger than hidden_size (NH) when model is pruned.
  auto gemm_data = allocator->Alloc(SafeInt<size_t>(batch_size) * sequence_length * 3 * hidden_size * element_size);
  BufferUniquePtr gemm_buffer(gemm_data, BufferDeleter(allocator));

  auto Q = reinterpret_cast<T*>(gemm_data);
  auto K = Q + static_cast<size_t>(batch_size) * sequence_length * hidden_size;
  auto V = K + static_cast<size_t>(batch_size) * sequence_length * hidden_size;
  T* QKV[3] = {Q, K, V};

  {
    const int loop_len = 3 * batch_size * num_heads_;
    const auto* input_data = input->template Data<T>();
    const auto* weights_data = weights ? weights->template Data<T>() : nullptr;
    const auto* bias_data = bias->template Data<T>();

    const double cost =
        static_cast<double>(sequence_length) * static_cast<double>(head_size) * static_cast<double>(input_hidden_size);
    ThreadPool::TryParallelFor(tp, loop_len, cost, [&](std::ptrdiff_t begin, std::ptrdiff_t end) {
      for (std::ptrdiff_t i = begin; i != end; ++i) {
        const int batch_index = static_cast<int>((i / 3) / num_heads_);
        const int head_index = static_cast<int>((i / 3) % num_heads_);
        const int qkv_index = static_cast<int>(i % 3); // 041

        int input_offset = batch_index * sequence_length * input_hidden_size;
        int weights_offset = qkv_index * hidden_size + head_index * head_size;
        T* qkv_dest = QKV[qkv_index];
        int qkv_offset = (batch_index * num_heads_ + head_index) * (sequence_length * head_size); // 0 1024 32768

        // TODO!! memcpy here makes it not worthwhile to use Gemm batch. Possible to post process?
        // broadcast 3NH -> (3.B.N.S.H)
        const T* broadcast_data_src = bias_data + weights_offset;
        T* broadcast_data_dest = QKV[qkv_index] + qkv_offset;
        for (int seq_index = 0; seq_index < sequence_length; seq_index++) {
          memcpy(broadcast_data_dest, broadcast_data_src, head_size * sizeof(T));
          broadcast_data_dest += head_size;
        }
        // 128 64 768, 0, 0.10000001, 768, 0.000778210117 196608, 0.00011189028 32768, 64
        //                   original           transposed            iteration
        // A: input          (BxSxD)            (B.)S x D             S x D
        // B: weights        (Dx3xNxH)          D x (3.N.)H           D x H
        // C: QKV[qkv_index] (3xBxNxSxH)        (3.B.N.)S x H         S x H
        if (packed_weights_) {
          const auto* packed_weight =
              static_cast<const uint8_t*>(packed_weights_.get()) + packed_weights_size_ * (weights_offset / head_size);
          MlasGemm(
              CblasNoTrans,               // TransA = no
              sequence_length,            // M      = S
              head_size,                  // N      = H
              input_hidden_size,          // K      = D
              1.0f,                       // alpha
              input_data + input_offset,  // A
              input_hidden_size,          // lda    = D
              packed_weight,              // B
              1.0f,                       // beta
              qkv_dest + qkv_offset,      // C
              head_size,                  // ldc
              nullptr);                   // use single-thread
        } else {
          math::GemmEx<float, ThreadPool>(
              CblasNoTrans,                   // TransA = no
              CblasNoTrans,                   // TransB = no
              sequence_length,                // M      = S
              head_size,                      // N      = H
              input_hidden_size,              // K      = D
              1.0f,                           // alpha
              input_data + input_offset,      // A
              input_hidden_size,              // lda    = D
              weights_data + weights_offset,  // B
              3 * hidden_size,                // ldb    = 3NH
              1.0f,                           // beta
              qkv_dest + qkv_offset,          // C
              head_size,                      // ldc
              nullptr                         // use single-thread
          );
        }
      }
    });
  }

  // Compute the attention score and apply the score to V
  return ApplyAttention(Q, K, V, mask_index, past, output,
                        batch_size, sequence_length,
                        head_size, hidden_size, context);
}

// TODO: currently only for simple self attention without previous frames result
template<>
Status Attention<float>::Compute(OpKernelContext* context) const {
    const Tensor* input = context->Input<Tensor>(0);
  const Tensor* weights = packed_weights_ ? nullptr : context->Input<Tensor>(1);
  const Tensor* bias = context->Input<Tensor>(2);
  const Tensor* mask_index = context->Input<Tensor>(3);
  const Tensor* past = context->Input<Tensor>(4);

  const TensorShape& weights_shape = (weights ? weights->Shape() : weight_shape_);
  ORT_RETURN_IF_ERROR(CheckInputs(input->Shape(),
                                  weights_shape,
                                  bias->Shape(),
                                  mask_index,
                                  past));

  const auto& shape = input->Shape().GetDims();
  const int batch_size = static_cast<int>(shape[0]);
  const int sequence_length = static_cast<int>(shape[1]);
  const int input_hidden_size = static_cast<int>(shape[2]);

  const auto& weights_dims = weights_shape.GetDims();
  const int hidden_size = static_cast<int>(weights_dims[1]) / 3;
  const int head_size = hidden_size / num_heads_;

  std::vector<int64_t> output_shape(3);
  output_shape[0] = shape[0];
  output_shape[1] = shape[1];
  output_shape[2] = static_cast<int64_t>(hidden_size);
  Tensor* output = context->Output(0, output_shape);

  constexpr size_t element_size = sizeof(float);

  AllocatorPtr allocator;
  ORT_RETURN_IF_ERROR(context->GetTempSpaceAllocator(&allocator));

  auto* tp = context->GetOperatorThreadPool();
  // Compute Q, K, V
  // gemm_data(BS, 3NH) = input(BS, D) x weights(D, 3NH) + bias(3NH)
  // D (input_hidden_size) is hidden dimension of input, where D could be larger than hidden_size (NH) when model is pruned.
  auto gemm_data = allocator->Alloc(SafeInt<size_t>(batch_size) * sequence_length * 3 * hidden_size * element_size);
  BufferUniquePtr gemm_buffer(gemm_data, BufferDeleter(allocator));

  auto Q = reinterpret_cast<float*>(gemm_data);
  auto K = Q + static_cast<size_t>(batch_size) * sequence_length * hidden_size;
  auto V = K + static_cast<size_t>(batch_size) * sequence_length * hidden_size;
  float* QKV[3] = {Q, K, V};

  {
    const int loop_len = 3 * batch_size * num_heads_;
    const auto* input_data = input->template Data<float>();
    const auto* weights_data = weights ? weights->template Data<float>() : nullptr;
    const auto* bias_data = bias->template Data<float>();

    const double cost =
        static_cast<double>(sequence_length) * static_cast<double>(head_size) * static_cast<double>(input_hidden_size);
    ThreadPool::TryParallelFor(tp, loop_len, cost, [&](std::ptrdiff_t begin, std::ptrdiff_t end) {
      for (std::ptrdiff_t i = begin; i != end; ++i) {
        const int batch_index = static_cast<int>((i / 3) / num_heads_);
        const int head_index = static_cast<int>((i / 3) % num_heads_);
        const int qkv_index = static_cast<int>(i % 3); // 041

        int input_offset = batch_index * sequence_length * input_hidden_size;
        int weights_offset = qkv_index * hidden_size + head_index * head_size;
        float* qkv_dest = QKV[qkv_index];
        int qkv_offset = (batch_index * num_heads_ + head_index) * (sequence_length * head_size); // 0 1024 32768

        //// CHANGE6: QKV bias
        //// TODO!! memcpy here makes it not worthwhile to use Gemm batch. Possible to post process?
        //// broadcast 3NH -> (3.B.N.S.H)
        //const float* broadcast_data_src = bias_data + weights_offset;
        //float* broadcast_data_dest = QKV[qkv_index] + qkv_offset;
        //for (int seq_index = 0; seq_index < sequence_length; seq_index++) {
        //  memcpy(broadcast_data_dest, broadcast_data_src, head_size * sizeof(float));
        //  broadcast_data_dest += head_size;
        //}

        if (packed_weights_) {
          const auto* packed_weight =
              static_cast<const uint8_t*>(packed_weights_.get()) + packed_weights_size_ * (weights_offset / head_size);

          //MlasGemm(
          //    CblasNoTrans,               // TransA = no
          //    sequence_length,            // M      = S
          //    head_size,                  // N      = H
          //    input_hidden_size,          // K      = D
          //    1.0f,                       // alpha
          //    input_data + input_offset,  // A
          //    input_hidden_size,          // lda    = D
          //    packed_weight,              // B
          //    1.0f,                       // beta
          //    qkv_dest + qkv_offset,      // C
          //    head_size,                  // ldc
          //    nullptr);                   // use single-thread
          // CHANGE6: QKV bias
          MLAS_SGEMM_DATA_PARAMS dp;dp.BIsPacked = true;
          dp.A = input_data + input_offset; dp.lda = input_hidden_size;dp.alpha = 1.0f;
          dp.B = reinterpret_cast<const float*>(packed_weight);dp.ldb = 0;dp.beta = 1.0f;
          dp.C = qkv_dest + qkv_offset;dp.ldc = head_size;
          dp.Bias = bias_data + weights_offset;
          MlasGemmBatchKN(CblasNoTrans, CblasTrans, sequence_length, head_size, input_hidden_size, &dp, 1, nullptr,
            768, 128, 1, sequence_length >= head_size);

          //// CHANGE1 : StrideK
          //MLAS_SGEMM_DATA_PARAMS dp;dp.BIsPacked = true;
					//dp.A = input_data + input_offset; dp.lda = input_hidden_size;dp.alpha = 1.0f;
					//dp.B = reinterpret_cast<const float*>(packed_weight);dp.ldb = 0;dp.beta = 1.0f;
					//dp.C = qkv_dest + qkv_offset;dp.ldc = head_size;
					//MlasGemmBatchKN(CblasNoTrans, CblasTrans, sequence_length, head_size, input_hidden_size, &dp, 1, nullptr,
					//	768, 128, 1, sequence_length >= head_size);
        } else {
          math::GemmEx<float, ThreadPool>(
              CblasNoTrans,                   // TransA = no
              CblasNoTrans,                   // TransB = no
              sequence_length,                // M      = S
              head_size,                      // N      = H
              input_hidden_size,              // K      = D
              1.0f,                           // alpha
              input_data + input_offset,      // A
              input_hidden_size,              // lda    = D
              weights_data + weights_offset,  // B
              3 * hidden_size,                // ldb    = 3NH
              1.0f,                           // beta
              qkv_dest + qkv_offset,          // C
              head_size,                      // ldc
              nullptr                         // use single-thread
          );
        }
      }
    });
  }

  //// Compute the attention score and apply the score to V
  //return ApplyAttention<float>(Q, K, V, mask_index, past, output,
  //                      batch_size, sequence_length,
  //                      head_size, hidden_size, context);
  
  // Expand
  int past_sequence_length = 0;
  //Tensor* present = GetPresent(context, past, batch_size, head_size, sequence_length, past_sequence_length);

  // Total sequence length including that of past state: S* = S' + S
  const int all_sequence_length = past_sequence_length + sequence_length;

  // Compute the attention score. It does 2 things:
  //         I. attention_probs(B, N, S, S*) = 1/sqrt(H) x Q(B, N, S, H) x K'(B, N, S*, H -> B, N, H, S*) +
  //                                           1 x mask_data(B, N, S, S*)
  //         II.attention_probs(B, N, S, S*) = Softmax(attention_probs)
  size_t attention_probs_bytes = SafeInt<size_t>(batch_size) * num_heads_ * sequence_length * all_sequence_length * sizeof(float);
  auto attention_probs = allocator->Alloc(attention_probs_bytes);
  BufferUniquePtr scratch_buffer(attention_probs, BufferDeleter(allocator));

  // Init Mask
  void* mask_data = nullptr;
  if (mask_index != nullptr || (is_unidirectional_ && sequence_length > 1)) {
    size_t mask_data_bytes = SafeInt<size_t>(batch_size) * sequence_length * all_sequence_length * sizeof(float);
    mask_data = allocator->Alloc(mask_data_bytes);
    memset(mask_data, 0, mask_data_bytes);
  }
  BufferUniquePtr mask_data_buffer(mask_data, BufferDeleter(allocator));
  const int32_t* mask_index_data = mask_index != nullptr ? mask_index->template Data<int32_t>() : nullptr;
  const std::vector<int64_t>* mask_index_dims = mask_index != nullptr ? &(mask_index->Shape().GetDims()) : nullptr;

  //const T* past_data = past != nullptr ? past->template Data<T>() : nullptr;
  //T* present_data = present != nullptr ? present->template MutableData<T>() : nullptr;

  // mask_index --> mask_index_data, convert attention_probs, mask_data
  //ComputeAttentionProbs<float>(static_cast<float*>(attention_probs), Q, K,
  //                         mask_index_data, mask_index_dims, static_cast<float*>(mask_data),
  //                         batch_size, sequence_length, past_sequence_length, head_size,
  //                         nullptr, nullptr, tp);
  
  const size_t input_chunk_length = static_cast<size_t>(sequence_length) * head_size;      // S x H
  //const size_t past_chunk_length = static_cast<size_t>(past_sequence_length) * head_size;  // S' x H
  //const size_t present_chunk_length = past_chunk_length + input_chunk_length;              // S* x H

  {
    if (mask_data != nullptr) {
      PrepareMask(mask_index_data, mask_index_dims, static_cast<float*>(mask_data), is_unidirectional_, batch_size, sequence_length, past_sequence_length);
    } else {  // no any mask
      memset(attention_probs, 0, static_cast<size_t>(batch_size) * num_heads_ * sequence_length * all_sequence_length * sizeof(float));
    }

    const int loop_len = batch_size * num_heads_;
    const float alpha = 1.0f / sqrt(static_cast<float>(head_size));

    // The cost of Gemm
    const double cost = static_cast<double>(head_size) * sequence_length * all_sequence_length;

    float* output_data = output->template MutableData<float>();
    //// CHANGE3: VxAtt
    //auto out_tmp_data =
    //  allocator->Alloc(SafeInt<size_t>(batch_size) * num_heads_ * sequence_length * head_size * sizeof(float));
    //BufferUniquePtr out_tmp_buffer(out_tmp_data, BufferDeleter(allocator));
    //float* tmp_buffer = static_cast<float*>(out_tmp_data);


    ThreadPool::TryParallelFor(tp, loop_len, cost, [&](std::ptrdiff_t begin, std::ptrdiff_t end) {
      for (std::ptrdiff_t i = begin; i != end; ++i) {
        const std::ptrdiff_t batch_index = i / num_heads_;

        MLAS_SGEMM_DATA_PARAMS dp;

        // CHANGE5: MMM mask
        // broadcast mask data: (Bx)SxS* -> (BxNx)SxS*
        if (mask_data != nullptr) {
          const float* broadcast_data_src = reinterpret_cast<float*>(mask_data) + batch_index * sequence_length * all_sequence_length;
          float* broadcast_data_dest = reinterpret_cast<float*>(attention_probs) + sequence_length * all_sequence_length * i;
          memcpy(broadcast_data_dest, broadcast_data_src, sequence_length * all_sequence_length * sizeof(float));
        }
        const float* k = K + input_chunk_length * i;
        math::Gemm<float, ThreadPool>(CblasNoTrans, CblasTrans, sequence_length, all_sequence_length, head_size, alpha,
                                  Q + input_chunk_length * i, k, 1.0,
                                  reinterpret_cast<float*>(attention_probs) + sequence_length * all_sequence_length * i, nullptr);
        // CHANGE2: softmax
        const int sN = sequence_length;
        const int sD = all_sequence_length;
        MlasComputeSoftmax(
          reinterpret_cast<float*>(attention_probs) + sequence_length * all_sequence_length * i,
          reinterpret_cast<float*>(attention_probs) + sequence_length * all_sequence_length * i,
          sN, sD, false, nullptr);

        //// CHANGE3: VxAtt
        //const float* v = V + input_chunk_length * i;
        //float* current_tmp_data = reinterpret_cast<float*>(tmp_buffer) + input_chunk_length * i;
        //math::MatMul<float>(sequence_length, head_size, all_sequence_length,
        //                static_cast<float*>(attention_probs) + sequence_length * all_sequence_length * i,
        //                v, current_tmp_data, nullptr);
        //const int head_index = static_cast<int>(i % num_heads_);
        //float* src = current_tmp_data;
        //float* dest = output_data + (batch_index * sequence_length * num_heads_ + head_index) * head_size;
        //const auto bytes_to_copy = SafeInt<size_t>(head_size) * sizeof(float);
        //for (int j = 0; j < sequence_length; j++) {
        //  memcpy(dest, src, bytes_to_copy);
        //  src += head_size;
        //  dest += hidden_size;
        //}

        // CHANGE4: save res mem
        dp.A = static_cast<float*>(attention_probs) + sequence_length * all_sequence_length * i;dp.lda = all_sequence_length;dp.alpha = 1.f;
        dp.B = V + all_sequence_length*head_size*i;dp.ldb = head_size;dp.beta = 0.f;
        const int head_index = static_cast<int>(i % num_heads_);
        dp.C = output_data + (batch_index * sequence_length * num_heads_ + head_index) * head_size;dp.ldc = hidden_size;
        MlasGemmBatch(CblasNoTrans, CblasNoTrans, sequence_length, head_size, all_sequence_length, &dp, 1, nullptr);
      }
    });
  }

  // CHANGE2: softmax
  //  attention_probs(B, N, S, S*) = Softmax(attention_probs)
  //{
  //  const int N = batch_size * num_heads_ * sequence_length;
  //  const int D = all_sequence_length;
  //  ComputeAttentionSoftmaxInplace(static_cast<float*>(attention_probs), N, D, tp);
  //}


  // CHANGE3: VxAtt
  //// Compute the attentionScore * Value. It does: out_tmp(B, N, S, H) = attention_probs(B, N, S, S*) x V(B, N, S*, H)
  //auto out_tmp_data =
  //    allocator->Alloc(SafeInt<size_t>(batch_size) * num_heads_ * sequence_length * head_size * sizeof(float));
  //BufferUniquePtr out_tmp_buffer(out_tmp_data, BufferDeleter(allocator));


  //// output --> output_data, convert attention_probs V 
  ////ComputeVxAttentionScore(output->template MutableData<float>(), static_cast<float*>(out_tmp_data), static_cast<const float*>(attention_probs), static_cast<const float*>(V),
  ////                        batch_size, sequence_length, past_sequence_length, head_size, hidden_size,
  ////                        nullptr, nullptr, tp);
  //float* output_data = output->template MutableData<float>();
  //float* tmp_buffer = static_cast<float*>(out_tmp_data);
  
  ////const size_t input_chunk_length = static_cast<size_t>(sequence_length * head_size);      // S x H
  ////const size_t past_chunk_length = static_cast<size_t>(past_sequence_length * head_size);  // S' x H
  ////const size_t present_chunk_length = past_chunk_length + input_chunk_length;              // S* x H

  //// Move the pointer of past and present to start of v values.
  ////if (nullptr != past) {
  ////  past += batch_size * num_heads_ * past_sequence_length * head_size;
  ////}
  ////if (nullptr != present) {
  ////  present += batch_size * num_heads_ * all_sequence_length * head_size;
  ////}

  //const double cost =
  //    static_cast<double>(sequence_length) * static_cast<double>(head_size) * static_cast<double>(sequence_length);

  //ThreadPool::TryParallelFor(tp, batch_size * num_heads_, cost, [&](std::ptrdiff_t begin, std::ptrdiff_t end) {
  //  for (std::ptrdiff_t i = begin; i != end; ++i) {
  //    const float* v = V + input_chunk_length * i;
  //    //if (nullptr != present) {
  //    //  // concatenate past_V and V: (BxNx)S'xH, (BxNx)SxH -> (BxNx)S*xH
  //    //  v = ConcatStateChunk(past, v, present, past_chunk_length, present_chunk_length, i);
  //    //}

  //    float* current_tmp_data = reinterpret_cast<float*>(tmp_buffer) + input_chunk_length * i;
  //    math::MatMul<float>(sequence_length, head_size, all_sequence_length,
  //                    static_cast<float*>(attention_probs) + sequence_length * all_sequence_length * i,
  //                    v, current_tmp_data, nullptr);

  //    // transpose: out(B, S, N, H) = transpose out_tmp(B, N, S, H)
  //    const int batch_index = static_cast<int>(i / num_heads_);
  //    const int head_index = static_cast<int>(i % num_heads_);
  //    float* src = current_tmp_data;
  //    float* dest = output_data + (batch_index * sequence_length * num_heads_ + head_index) * head_size;
  //    const auto bytes_to_copy = SafeInt<size_t>(head_size) * sizeof(float);
  //    for (int j = 0; j < sequence_length; j++) {
  //      memcpy(dest, src, bytes_to_copy);
  //      src += head_size;
  //      dest += hidden_size;
  //    }
  //  }
  //});

  return Status::OK();
}

}  // namespace contrib
}  // namespace onnxruntime
