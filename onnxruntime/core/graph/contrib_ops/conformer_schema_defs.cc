// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/tensorprotoutils.h"
#include "core/graph/constants.h"
#include "core/graph/contrib_ops/contrib_defs.h"
#include "core/mlas/inc/mlas.h"

namespace onnxruntime {
namespace contrib {

using ONNX_NAMESPACE::AttributeProto;
using ONNX_NAMESPACE::InferenceContext;
using ONNX_NAMESPACE::OpSchema;
using ONNX_NAMESPACE::OPTIONAL_VALUE;

void matmulBiasShapeInference(
    ONNX_NAMESPACE::InferenceContext& ctx,
    int input1Idx,
    int input2Idx, int input3Idx) {
  if (!hasInputShape(ctx, input1Idx) || !hasInputShape(ctx, input2Idx)) {
    if(!hasInputShape(ctx, input3Idx)) // dummy
      return;
    return;
  }

  const auto shape0 = ctx.getInputType(input1Idx)->tensor_type().shape();
  const auto shape1 = ctx.getInputType(input2Idx)->tensor_type().shape();

  if (shape0.dim_size() == 0 || shape1.dim_size() == 0) {
    fail_shape_inference("Input tensors of wrong rank (0).");
  }

  ONNX_NAMESPACE::TensorShapeProto shapeL, shapeR;

  // First promote each shape to at least rank-2. This logic is
  // specific to matmul, not generic broadcasting.
  {
    if (shape0.dim_size() == 1) {
      shapeL.add_dim()->set_dim_value(1);
      *shapeL.add_dim() = shape0.dim(0);
    } else {
      *shapeL.mutable_dim() = shape0.dim();
    }
    if (shape1.dim_size() == 1) {
      *shapeR.add_dim() = shape1.dim(0);
      shapeR.add_dim()->set_dim_value(1);
    } else {
      *shapeR.mutable_dim() = shape1.dim();
    }
  }

  // Check for compatible matrix multiply dimensions
  {
    auto dimL = shapeL.dim(shapeL.dim_size() - 1);
    auto dimR = shapeR.dim(shapeR.dim_size() - 2);
    if (dimL.has_dim_value() && dimR.has_dim_value() &&
        dimL.dim_value() != dimR.dim_value()) {
      fail_shape_inference("Incompatible dimensions for matrix multiplication");
    }
  }

  ONNX_NAMESPACE::TensorShapeProto resultShape;

  // Now call out to generic multidimensional broadcasting for
  // the broadcastable prefixes.
  {
    ONNX_NAMESPACE::TensorShapeProto prefixShapeL, prefixShapeR;
    for (int i = 0; i < shapeL.dim_size() - 2; ++i) {
      *prefixShapeL.add_dim() = shapeL.dim(i);
    }
    for (int i = 0; i < shapeR.dim_size() - 2; ++i) {
      *prefixShapeR.add_dim() = shapeR.dim(i);
    }
    bidirectionalBroadcastShapeInference(
        prefixShapeL, prefixShapeR, resultShape);
  }

  // Back to matmul-specific. Add the trailing dimensions back in.
  {
    if (shape0.dim_size() != 1) {
      *resultShape.add_dim() = shapeL.dim(shapeL.dim_size() - 2);
    }
    if (shape1.dim_size() != 1) {
      *resultShape.add_dim() = shapeR.dim(shapeR.dim_size() - 1);
    }
  }

  *ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape() = resultShape;
}

//void matmulBiasShapeInference(
//    ONNX_NAMESPACE::InferenceContext& ctx,
//    int input1Idx,
//    int input2Idx,
//    int input3Idx) {
//  // require both input has shape info
//  if (!hasInputShape(ctx, input1Idx) || !hasInputShape(ctx, input2Idx) || !hasInputShape(ctx, input3Idx)) {
//    return;
//  }
//
//  const auto shape0 = ctx.getInputType(input1Idx)->tensor_type().shape();
//  const auto shape1 = ctx.getInputType(input2Idx)->tensor_type().shape();
//  const auto shape2 = ctx.getInputType(input3Idx)->tensor_type().shape();
//
//  if (shape0.dim_size() == 0 || shape1.dim_size() == 0) {
//    fail_shape_inference("Input tensors of wrong rank (0).");
//  }
//
//  ONNX_NAMESPACE::TensorShapeProto shapeL, shapeR, shapeB;
//
//  // Bias must be a vector with 1 dim
//  {
//    if (shape2.dim_size() != 1){
//      fail_shape_inference("Incompatible dimensions for matrix multiplication bias");
//    }
//  }
//
//  // First promote each shape to at least rank-2. This logic is
//  // specific to matmul, not generic broadcasting.
//  {
//    if (shape0.dim_size() == 1) {
//      shapeL.add_dim()->set_dim_value(1);
//      *shapeL.add_dim() = shape0.dim(0);
//    } else {
//      *shapeL.mutable_dim() = shape0.dim();
//    }
//    if (shape1.dim_size() == 1) {
//      *shapeR.add_dim() = shape1.dim(0);
//      shapeR.add_dim()->set_dim_value(1);
//    } else {
//      *shapeR.mutable_dim() = shape1.dim();
//    }
//    *shapeB.mutable_dim() = shape2.dim();
//  }
//
//  // Check for compatible matrix multiply dimensions, A last and B second last
//  // also there is possibility that it !has_dim_value, which means a dim_param
//  {
//    auto dimL = shapeL.dim(shapeL.dim_size() - 1);
//    auto dimR = shapeR.dim(shapeR.dim_size() - 2);
//    auto dimN = shapeR.dim(shapeR.dim_size() - 1);
//    auto dimB = shapeB.dim(0);
//    if (dimL.has_dim_value() && dimR.has_dim_value() &&
//        dimL.dim_value() != dimR.dim_value()) {
//      fail_shape_inference("Incompatible dimensions for matrix multiplication");
//    }
//    if (dimN.has_dim_value() && dimB.has_dim_value() &&
//        dimN.dim_value() != dimB.dim_value()) {
//      fail_shape_inference("Incompatible dimensions for matrix multiplication");
//    }
//  }
//
//  ONNX_NAMESPACE::TensorShapeProto resultShape;
//
//  // Now call out to generic multidimensional broadcasting for
//  // the broadcastable prefixes.
//  {
//    ONNX_NAMESPACE::TensorShapeProto prefixShapeL, prefixShapeR;
//    for (int i = 0; i < shapeL.dim_size() - 2; ++i) {
//      *prefixShapeL.add_dim() = shapeL.dim(i);
//    }
//    for (int i = 0; i < shapeR.dim_size() - 2; ++i) {
//      *prefixShapeR.add_dim() = shapeR.dim(i);
//    }
//    // infer from first [dim-2] dims of A and B
//    // what this functions do?
//    // 1. check [dim-2] dims of A and B whether they are the same, or one is 1 so that broadcast
//    // 2. for symbolic+value, set to value; for symbolic+same symbolic, set to symbolic; otherwise just add_dim()
//    // the output shape dim size is the largest of A and B
//    bidirectionalBroadcastShapeInference(
//        prefixShapeL, prefixShapeR, resultShape);
//  }
//
//  // Back to matmul-specific. Add the trailing dimensions back in.
//  {
//    if (shape0.dim_size() != 1) {
//      *resultShape.add_dim() = shapeL.dim(shapeL.dim_size() - 2);
//    }
//    if (shape1.dim_size() != 1) {
//      *resultShape.add_dim() = shapeR.dim(shapeR.dim_size() - 1);
//    }
//  }
//
//  *ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape() = resultShape;
//}

static const char* MatMulBias_doc = R"DOC(
Matrix product that behaves like numpy.matmul, but with bias support: https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.matmul.html
)DOC";;

void RegisterConformerSchemas() {
  ONNX_CONTRIB_OPERATOR_SCHEMA(MatMulBias)
        .SetDomain(kMSDomain)
        .SinceVersion(1)
        .SetDoc(MatMulBias_doc)
        .Input(
            0,
            "A",
            "Input tensor A. "
            "The shape of A should be (M1, M2, ..., K) if transA is 0, "
            "or (K, M) if transA is non-zero.",
            "T",
            OpSchema::Single,
            true,
            1, // min arity only usefull in Variadic
            OpSchema::NonDifferentiable) // haven't study on training
        .Input(
            1,
            "B",
            "Input tensor B. "
            "The shape of B should be (K, N1, N2...) if transB is 0, "
            "or (N, K) if transB is non-zero.",
            "T",
            OpSchema::Single,
            true,
            1,
            OpSchema::NonDifferentiable)
        .Input(
            2,
            "Bias",
            "input tensor Bias. "
            "The shape of C should be unidirectional broadcastable to (M, N).",
            "T",
            OpSchema::Single,
            true,
            1,
            OpSchema::NonDifferentiable)
        .Output(0,
            "Y",
            "Output tensor of shape (M, N).",
            "T",
            OpSchema::Single,
            true,
            1,
            OpSchema::NonDifferentiable)
        .TypeConstraint(
            "T",
            {"tensor(float16)",
             "tensor(float)",
             "tensor(double)",
             "tensor(uint32)",
             "tensor(uint64)",
             "tensor(int32)",
             "tensor(int64)",
             "tensor(bfloat16)"},
            "Constrain input and output types to float/int tensors.")
        .Attr(
            "transA",
            "Whether A should be transposed, currently not supported",
            AttributeProto::INT,
            static_cast<int64_t>(0))
        .Attr(
            "transB",
            "Whether B should be transposed, currently not supported",
            AttributeProto::INT,
            static_cast<int64_t>(0))
        .Attr(
            "alpha",
            "Scalar multiplier for the product of input tensors A * B.",
            AttributeProto::FLOAT,
            1.0f)
        .Attr(
            "beta",
            "Scalar multiplier for input tensor Bias. Currently only can be 1.0",
            AttributeProto::FLOAT,
            1.0f)
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          propagateElemTypeFromInputToOutput(ctx, 0, 0); // propogate the type of the first input to output
          matmulBiasShapeInference(ctx, 0, 1, 2);
        });

}

}  // namespace contrib
}  // namespace onnxruntime
