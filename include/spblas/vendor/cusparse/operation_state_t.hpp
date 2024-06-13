#pragma once

#include <memory>
#include <type_traits>

#include <cuda_runtime.h>
#include <cusparse.h>

#include <spblas/backend/concepts.hpp>
#include <spblas/detail/operation_info_t.hpp>
#include <spblas/detail/ranges.hpp>
#include <spblas/detail/view_inspectors.hpp>

#include "types.hpp"

namespace spblas {

class spmv_handle_t {
public:
  spmv_handle_t(std::shared_ptr<const allocator> alloc)
      : alloc_(alloc), buffer_size_(0) {
    cusparseCreate(&handle_);
  }

  ~spmv_handle_t() {
    alloc_->free(workspace_);
    cusparseDestroy(handle_);
  }

  template <matrix A, vector B, vector C>
    requires __detail::has_csr_base<A> &&
             __detail::has_contiguous_range_base<B> &&
             __ranges::contiguous_range<C>
  void multiply(A&& a, B&& b, C&& c) {
    auto a_base = __detail::get_ultimate_base(a);
    auto b_base = __detail::get_ultimate_base(b);
    using matrix_type = decltype(a_base);
    using input_type = decltype(b_base);
    using output_type = std::remove_reference_t<decltype(c)>;
    using value_type = typename matrix_type::scalar_type;

    auto alpha_optional = __detail::get_scaling_factor(a, b);
    tensor_scalar_t<A> alpha = alpha_optional.value_or(1);

    cusparseSpMatDescr_t mat;
    cusparseCreateCsr(&mat, __backend::shape(a_base)[0],
                      __backend::shape(a_base)[0], a_base.values().size(),
                      a_base.rowptr().data(), a_base.colind().data(),
                      a_base.values().data(),
                      cusparse_index_type<typename matrix_type::offset_type>(),
                      cusparse_index_type<typename matrix_type::index_type>(),
                      CUSPARSE_INDEX_BASE_ZERO, cuda_data_type<value_type>());

    cusparseDnVecDescr_t vecb, vecc;
    cusparseCreateDnVec(&vecb, b_base.size(), b_base.data(),
                        cuda_data_type<typename input_type::value_type>());
    cusparseCreateDnVec(&vecc, c.size(), c.data(),
                        cuda_data_type<typename output_type::value_type>());

    value_type alpha_val = alpha;
    value_type beta = 0.0;
    long unsigned int buffer_size = 0;
    // TODO: create a compute type for mixed precision computation
    cusparseSpMV_bufferSize(handle_, CUSPARSE_OPERATION_NON_TRANSPOSE,
                            &alpha_val, mat, vecb, &beta, vecc,
                            cuda_data_type<value_type>(),
                            CUSPARSE_SPMV_ALG_DEFAULT, &buffer_size);
    // only allocate the new workspace when the requiring workspace larger than
    // current
    if (buffer_size > buffer_size_) {
      buffer_size_ = buffer_size;
      alloc_->free(workspace_);
      alloc_->alloc(&workspace_, buffer_size);
    }

    cusparseSpMV(handle_, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha_val, mat,
                 vecb, &beta, vecc, cuda_data_type<value_type>(),
                 CUSPARSE_SPMV_ALG_DEFAULT, workspace_);
    cudaDeviceSynchronize();
    cusparseDestroyDnVec(vecc);
    cusparseDestroyDnVec(vecb);
  }

private:
  cusparseHandle_t handle_;
  std::shared_ptr<const allocator> alloc_;
  long unsigned int buffer_size_;
  void* workspace_;
};

class spgemm_handle_t {
public:
  spgemm_handle_t(std::shared_ptr<const allocator> alloc)
      : alloc_(alloc), buffer_size1_(0), buffer_size2_(0), result_nnz_(0),
        result_shape_(0, 0) {
    cusparseCreate(&handle_);

    cusparseSpGEMM_createDescr(&spgemm_desc_);
  }

  ~spgemm_handle_t() {
    alloc_->free(workspace1_);
    alloc_->free(workspace2_);
    cusparseSpGEMM_destroyDescr(spgemm_desc_);
    cusparseDestroy(handle_);
  }

  auto result_shape() {
    return result_shape_;
  }

  auto result_nnz() {
    return result_nnz_;
  }

  template <matrix A, matrix B, matrix C>
    requires __detail::has_csr_base<A> && __detail::has_csr_base<B> &&
             __detail::is_csr_view_v<C>
  void multiply_compute(A&& a, B&& b, C&& c) {
    auto a_base = __detail::get_ultimate_base(a);
    auto b_base = __detail::get_ultimate_base(b);
    using matrix_type = decltype(a_base);
    using input_type = decltype(b_base);
    using output_type = std::remove_reference_t<decltype(c)>;
    using value_type = typename matrix_type::scalar_type;

    cusparseSpMatDescr_t matA, matB, matC;
    size_t buffer_size1 = 0, buffer_size2 = 0;
    value_type alpha = 1.0;
    value_type beta = 0.0;
    // Create sparse matrix A in CSR format
    cusparseCreateCsr(&matA, __backend::shape(a_base)[0],
                      __backend::shape(a_base)[1], a_base.values().size(),
                      a_base.rowptr().data(), a_base.colind().data(),
                      a_base.values().data(),
                      cusparse_index_type<typename matrix_type::offset_type>(),
                      cusparse_index_type<typename matrix_type::index_type>(),
                      CUSPARSE_INDEX_BASE_ZERO, cuda_data_type<value_type>());
    cusparseCreateCsr(&matB, __backend::shape(b_base)[0],
                      __backend::shape(b_base)[1], b_base.values().size(),
                      b_base.rowptr().data(), b_base.colind().data(),
                      b_base.values().data(),
                      cusparse_index_type<typename input_type::offset_type>(),
                      cusparse_index_type<typename input_type::index_type>(),
                      CUSPARSE_INDEX_BASE_ZERO,
                      cuda_data_type<typename input_type::scalar_type>());
    cusparseCreateCsr(&matC, __backend::shape(a_base)[0],
                      __backend::shape(b_base)[1], 0, c.rowptr().data(), NULL,
                      NULL,
                      cusparse_index_type<typename output_type::offset_type>(),
                      cusparse_index_type<typename output_type::index_type>(),
                      CUSPARSE_INDEX_BASE_ZERO,
                      cuda_data_type<typename output_type::scalar_type>());
    //--------------------------------------------------------------------------
    // SpGEMM Computation
    // ask buffer_size1 bytes for external memory
    cusparseSpGEMM_workEstimation(
        handle_, CUSPARSE_OPERATION_NON_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, matB, &beta, matC,
        cuda_data_type<value_type>(), CUSPARSE_SPGEMM_DEFAULT, spgemm_desc_,
        &buffer_size1, NULL);
    if (buffer_size1 > buffer_size1_) {
      buffer_size1_ = buffer_size1;
      alloc_->free(workspace1_);
      alloc_->alloc(&workspace1_, buffer_size1);
    }
    // inspect the matrices A and B to understand the memory requirement for
    // the next step
    cusparseSpGEMM_workEstimation(
        handle_, CUSPARSE_OPERATION_NON_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, matB, &beta, matC,
        cuda_data_type<value_type>(), CUSPARSE_SPGEMM_DEFAULT, spgemm_desc_,
        &buffer_size1, workspace1_);

    // ask buffer_size2 bytes for external memory
    cusparseSpGEMM_compute(handle_, CUSPARSE_OPERATION_NON_TRANSPOSE,
                           CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, matB,
                           &beta, matC, cuda_data_type<value_type>(),
                           CUSPARSE_SPGEMM_DEFAULT, spgemm_desc_, &buffer_size2,
                           NULL);
    if (buffer_size2 > buffer_size2_) {
      buffer_size2_ = buffer_size2;
      alloc_->free(workspace2_);
      alloc_->alloc(&workspace2_, buffer_size2);
    }

    // compute the intermediate product of A * B
    cusparseSpGEMM_compute(handle_, CUSPARSE_OPERATION_NON_TRANSPOSE,
                           CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, matB,
                           &beta, matC, cuda_data_type<value_type>(),
                           CUSPARSE_SPGEMM_DEFAULT, spgemm_desc_, &buffer_size2,
                           workspace2_);
    // get matrix C non-zero entries C_nnz1
    int64_t C_num_rows1, C_num_cols1;
    cusparseSpMatGetSize(matC, &C_num_rows1, &C_num_cols1, &result_nnz_);
    result_shape_ = index<>(C_num_rows1, C_num_cols1);

    cusparseDestroySpMat(matA);
    cusparseDestroySpMat(matB);
    cusparseDestroySpMat(matC);
  }

  template <matrix A, matrix B, matrix C>
    requires __detail::has_csr_base<A> && __detail::has_csr_base<B> &&
             __detail::is_csr_view_v<C>
  void multiply_execute(A&& a, B&& b, C&& c) {
    auto a_base = __detail::get_ultimate_base(a);
    auto b_base = __detail::get_ultimate_base(b);
    using matrix_type = decltype(a_base);
    using input_type = decltype(b_base);
    using output_type = std::remove_reference_t<decltype(c)>;
    using value_type = typename matrix_type::scalar_type;

    auto alpha_optional = __detail::get_scaling_factor(a, b);
    tensor_scalar_t<A> alpha = alpha_optional.value_or(1);

    cusparseSpMatDescr_t matA, matB, matC;
    value_type alpha_val = alpha;
    value_type beta = 0.0;
    // Create sparse matrix A in CSR format
    cusparseCreateCsr(&matA, __backend::shape(a_base)[0],
                      __backend::shape(a_base)[1], a_base.values().size(),
                      a_base.rowptr().data(), a_base.colind().data(),
                      a_base.values().data(),
                      cusparse_index_type<typename matrix_type::offset_type>(),
                      cusparse_index_type<typename matrix_type::index_type>(),
                      CUSPARSE_INDEX_BASE_ZERO, cuda_data_type<value_type>());
    cusparseCreateCsr(&matB, __backend::shape(b_base)[0],
                      __backend::shape(b_base)[1], b_base.values().size(),
                      b_base.rowptr().data(), b_base.colind().data(),
                      b_base.values().data(),
                      cusparse_index_type<typename input_type::offset_type>(),
                      cusparse_index_type<typename input_type::index_type>(),
                      CUSPARSE_INDEX_BASE_ZERO,
                      cuda_data_type<typename input_type::scalar_type>());
    cusparseCreateCsr(&matC, __backend::shape(c)[0], __backend::shape(c)[1],
                      c.values().size(), c.rowptr().data(), c.colind().data(),
                      c.values().data(),
                      cusparse_index_type<typename output_type::offset_type>(),
                      cusparse_index_type<typename output_type::index_type>(),
                      CUSPARSE_INDEX_BASE_ZERO,
                      cuda_data_type<typename output_type::scalar_type>());

    cusparseSpGEMM_copy(handle_, CUSPARSE_OPERATION_NON_TRANSPOSE,
                        CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, matB,
                        &beta, matC, cuda_data_type<value_type>(),
                        CUSPARSE_SPGEMM_DEFAULT, spgemm_desc_);
    // destroy matrix/vector descriptors
    cudaDeviceSynchronize();
    cusparseDestroySpMat(matA);
    cusparseDestroySpMat(matB);
    cusparseDestroySpMat(matC);
  }

private:
  cusparseHandle_t handle_;
  cusparseSpGEMMDescr_t spgemm_desc_;
  std::shared_ptr<const allocator> alloc_;
  long unsigned int buffer_size1_;
  long unsigned int buffer_size2_;
  void* workspace1_;
  void* workspace2_;
  index<> result_shape_;
  index_t result_nnz_;
};

namespace __cusparse {

struct operation_state_t {
  cusparseSpGEMMDescr_t spgemm_descr;
};

} // namespace __cusparse

} // namespace spblas
