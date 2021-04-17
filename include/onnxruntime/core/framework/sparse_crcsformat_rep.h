// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/sparse_tensor.h"

namespace onnxruntime {
/// <summary>
/// This class represents Compressed Storage Row(Column) sparse format
/// CSS for Row Major Matrices
/// CSC for Column Major Matrices
/// </summary>
class SparseCrcsFormatRep : public SparseRep {
 public:
  enum Order {
    kRowMajor = 1,
    kColMajor = 2
  };

  /// <summary>
  /// This constructor allocates memory for indices using the provided allocator
  /// </summary>
  /// <param name="order">matrix order represented</param>
  /// <param name="inner_shape">inner indices shape. Normally 1-D, but we would like to support batches in the future</param>
  /// <param name="outer_shape">outer indices shape. Normally 1-D, but we would like to support batches in the future</param>
  /// <param name="allocator"></param>
  SparseCrcsFormatRep(Order order, const TensorShape& inner_shape,
                      const TensorShape& outer_shape, const AllocatorPtr& allocator)
      : major_(order),
        inner_indecies_(DataTypeImpl::GetType<int64_t>(), inner_shape, allocator),
        outer_indecies_(DataTypeImpl::GetType<int64_t>(), outer_shape, allocator) {}

  SparseCrcsFormatRep(Order order, const TensorShape& inner_shape, const TensorShape& outer_shape,
                      int64_t* inner_data, int64_t* outer_data,
                      const OrtMemoryInfo& info)
      : major_(order),
        inner_indecies_(DataTypeImpl::GetType<int64_t>(), inner_shape, inner_data, info, 0),
        outer_indecies_(DataTypeImpl::GetType<int64_t>(), outer_shape, outer_data, info, 0) {}

  ~SparseCrcsFormatRep() override;

  /// <summary>
  /// Returns the matrix order currently represented
  /// </summary>
  /// <returns></returns>
  Order Major() const noexcept {
    return major_;
  }

  const Tensor& Inner() const noexcept {
    return inner_indecies_;
  }

  const Tensor& Outer() const noexcept {
    return outer_indecies_;
  }

  Tensor& MutableInner() noexcept {
    return inner_indecies_;
  }

  Tensor& MutableOuter() noexcept {
    return outer_indecies_;
  }

  Status Copy(const DataTransferManager& data_transfer_manager, const AllocatorPtr& allocator,
              int exec_q_id, std::unique_ptr<SparseRep>& dst_rep) const override;

  Status Copy(const IDataTransfer& data_transfer, const AllocatorPtr& allocator,
              int exec_q_id, std::unique_ptr<SparseRep>& dst_rep) const override;

 private:
  Order major_;
  // This represents indices of nnz within a row/column
  Tensor inner_indecies_;
  // This represents indices into values and inner_indecies where each row/column data starts
  Tensor outer_indecies_;
};

class SparseCrcsBuilder {
 public:
  SparseCrcsBuilder(AllocatorPtr allocator, SparseTensor& sp, std::unique_ptr<SparseRep>& rep) noexcept
      : allocator_(std::move(allocator)),
        sp_(&sp),
        rep_(&rep) {}

  ~SparseCrcsBuilder() = default;

  Status GetOrCreate(SparseCrcsFormatRep::Order major,
                     const TensorShape& inner, const TensorShape& outer,
                     SparseCrcsFormatRep*& result);

  Status GetOrCreate(SparseCrcsFormatRep::Order major,
                     const TensorShape& inner, const TensorShape& outer,
                     int64_t* inner_data, int64_t* outer_data,
                     SparseCrcsFormatRep*& result);

 private:
  AllocatorPtr allocator_;
  SparseTensor* sp_;
  std::unique_ptr<SparseRep>* rep_;
};

template <>
inline const SparseCrcsFormatRep* SparseTensor::GetRep<SparseCrcsFormatRep>() const {
  ORT_ENFORCE(IsSet(format_flags_, SparseFormatFlags::kCrcs), "Expecting CSR or CRC format");
  return static_cast<const SparseCrcsFormatRep*>(rep_.get());
}

template <>
inline SparseCrcsBuilder SparseTensor::RepBuilder<SparseCrcsBuilder>() {
  if (!rep_) {
    format_flags_ = Set(format_flags_, SparseFormatFlags::kCrcs);
  } else {
    ORT_ENFORCE(IsSet(format_flags_, SparseFormatFlags::kCrcs), "Expecting CSR(C) format set");
  }
  return SparseCrcsBuilder(allocator_, *this, rep_);
}

}  // namespace onnxruntime
