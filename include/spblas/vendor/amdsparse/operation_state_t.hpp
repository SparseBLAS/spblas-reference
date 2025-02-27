#pragma once

#include <memory>
#include <type_traits>

#include <hip/hip_runtime.h>
#include <hipsparse/hipsparse.h>
#include <rocsparse/rocsparse.h>

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
    hipsparseCreate(&handle_);
  }

  ~spmv_handle_t() {
    alloc_->free(workspace_);
    hipsparseDestroy(handle_);
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

    hipsparseSpMatDescr_t mat;
    hipsparseCreateCsr(
        &mat, __backend::shape(a_base)[0], __backend::shape(a_base)[0],
        a_base.values().size(), a_base.rowptr().data(), a_base.colind().data(),
        a_base.values().data(),
        hipsparse_index_type<typename matrix_type::offset_type>(),
        hipsparse_index_type<typename matrix_type::index_type>(),
        HIPSPARSE_INDEX_BASE_ZERO, hip_data_type<value_type>());

    hipsparseDnVecDescr_t vecb, vecc;
    hipsparseCreateDnVec(&vecb, b_base.size(), b_base.data(),
                         hip_data_type<typename input_type::value_type>());
    hipsparseCreateDnVec(&vecc, c.size(), c.data(),
                         hip_data_type<typename output_type::value_type>());

    value_type alpha_val = alpha;
    value_type beta = 0.0;
    long unsigned int buffer_size = 0;
    // TODO: create a compute type for mixed precision computation
    hipsparseSpMV_bufferSize(handle_, HIPSPARSE_OPERATION_NON_TRANSPOSE,
                             &alpha_val, mat, vecb, &beta, vecc,
                             hip_data_type<value_type>(),
                             HIPSPARSE_SPMV_ALG_DEFAULT, &buffer_size);
    // only allocate the new workspace when the requiring workspace larger than
    // current
    if (buffer_size > buffer_size_) {
      buffer_size_ = buffer_size;
      alloc_->free(workspace_);
      alloc_->alloc(&workspace_, buffer_size);
    }

    hipsparseSpMV(handle_, HIPSPARSE_OPERATION_NON_TRANSPOSE, &alpha_val, mat,
                  vecb, &beta, vecc, hip_data_type<value_type>(),
                  HIPSPARSE_SPMV_ALG_DEFAULT, workspace_);
    hipDeviceSynchronize();
    hipsparseDestroyDnVec(vecc);
    hipsparseDestroyDnVec(vecb);
  }

private:
  hipsparseHandle_t handle_;
  std::shared_ptr<const allocator> alloc_;
  long unsigned int buffer_size_;
  void* workspace_;
};

class spgemm_handle_t {
public:
  spgemm_handle_t(std::shared_ptr<const allocator> alloc)
      : alloc_(alloc), buffer_size1_(0), result_nnz_(0), result_shape_(0, 0) {
    rocsparse_create_handle(&handle_);
  }

  ~spgemm_handle_t() {
    alloc_->free(workspace1_);
    rocsparse_destroy_handle(handle_);
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

    size_t buffer_size1 = 0;
    value_type alpha = 1.0;
    value_type beta = 0.0;
    // Create sparse matrix A in CSR format
    rocsparse_create_csr_descr(
        &matA, __backend::shape(a_base)[0], __backend::shape(a_base)[1],
        a_base.values().size(), a_base.rowptr().data(), a_base.colind().data(),
        a_base.values().data(),
        rocm_index_type<typename matrix_type::offset_type>(),
        rocm_index_type<typename matrix_type::index_type>(),
        rocsparse_index_base_zero, rocm_data_type<value_type>());
    rocsparse_create_csr_descr(
        &matB, __backend::shape(b_base)[0], __backend::shape(b_base)[1],
        b_base.values().size(), b_base.rowptr().data(), b_base.colind().data(),
        b_base.values().data(),
        rocm_index_type<typename input_type::offset_type>(),
        rocm_index_type<typename input_type::index_type>(),
        rocsparse_index_base_zero,
        rocm_data_type<typename input_type::scalar_type>());
    rocsparse_create_csr_descr(
        &matC, __backend::shape(a_base)[0], __backend::shape(b_base)[1], 0,
        c.rowptr().data(), NULL, NULL,
        rocm_index_type<typename output_type::offset_type>(),
        rocm_index_type<typename output_type::index_type>(),
        rocsparse_index_base_zero,
        rocm_data_type<typename output_type::scalar_type>());
    rocsparse_create_csr_descr(
        &matD, 0, 0, 0, nullptr, nullptr, nullptr,
        rocm_index_type<typename output_type::offset_type>(),
        rocm_index_type<typename output_type::index_type>(),
        rocsparse_index_base_zero,
        rocm_data_type<typename output_type::scalar_type>());
    //--------------------------------------------------------------------------
    // SpGEMM Computation
    // ask buffer_size1 bytes for external memory
    rocsparse_spgemm(
        handle_, rocsparse_operation_none, rocsparse_operation_none, &alpha,
        matA, matB, &beta, matD, matC, rocm_data_type<value_type>(),
        rocsparse_spgemm_alg_default, rocsparse_spgemm_stage_buffer_size,
        &buffer_size1, nullptr);
    if (buffer_size1 > buffer_size1_) {
      buffer_size1_ = buffer_size1;
      alloc_->free(workspace1_);
      alloc_->alloc(&workspace1_, buffer_size1);
    }
    rocsparse_spgemm(handle_, rocsparse_operation_none,
                     rocsparse_operation_none, &alpha, matA, matB, &beta, matD,
                     matC, rocm_data_type<value_type>(),
                     rocsparse_spgemm_alg_default, rocsparse_spgemm_stage_nnz,
                     &buffer_size1, workspace1_);
    // get matrix C non-zero entries C_nnz1
    int64_t C_num_rows1, C_num_cols1;
    rocsparse_spmat_get_size(matC, &C_num_rows1, &C_num_cols1, &result_nnz_);
    result_shape_ = index<>(C_num_rows1, C_num_cols1);
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
    
    value_type alpha_val = alpha;
    value_type beta = 0.0;
    rocsparse_csr_set_pointers(matC, c.rowptr().data(), c.colind().data(), c.values().data());

    rocsparse_spgemm(handle_, rocsparse_operation_none,
                     rocsparse_operation_none, &alpha, matA, matB, &beta, matD,
                     matC, rocm_data_type<value_type>(),
                     rocsparse_spgemm_alg_default, rocsparse_spgemm_stage_compute,
                     &buffer_size1_, workspace1_);
    hipDeviceSynchronize();
    // destroy matrix/vector descriptors
    rocsparse_destroy_spmat_descr(matA);
    rocsparse_destroy_spmat_descr(matB);
    rocsparse_destroy_spmat_descr(matC);
    rocsparse_destroy_spmat_descr(matD);
  }

  // split symbolic

  template <matrix A, matrix B, matrix C>
    requires __detail::has_csr_base<A> && __detail::has_csr_base<B> &&
             __detail::is_csr_view_v<C>
  void multiply_symbolic_fill(A&& a, B&& b, C&& c) {
    auto a_base = __detail::get_ultimate_base(a);
    auto b_base = __detail::get_ultimate_base(b);
    using matrix_type = decltype(a_base);
    using input_type = decltype(b_base);
    using output_type = std::remove_reference_t<decltype(c)>;
    using value_type = typename matrix_type::scalar_type;

    auto alpha_optional = __detail::get_scaling_factor(a, b);
    tensor_scalar_t<A> alpha = alpha_optional.value_or(1);
    
    value_type alpha_val = alpha;
    value_type beta = 0.0;
    rocsparse_csr_set_pointers(matC, c.rowptr().data(), c.colind().data(), c.values().data());

    rocsparse_spgemm(handle_, rocsparse_operation_none,
                     rocsparse_operation_none, &alpha, matA, matB, &beta, matD,
                     matC, rocm_data_type<value_type>(),
                     rocsparse_spgemm_alg_default, rocsparse_spgemm_stage_symbolic,
                     &buffer_size1_, workspace1_);
  }

  template <matrix A, matrix B, matrix C>
    requires __detail::has_csr_base<A> && __detail::has_csr_base<B> &&
             __detail::is_csr_view_v<C>
  void multiply_numeric(A&& a, B&& b, C&& c) {
    auto a_base = __detail::get_ultimate_base(a);
    auto b_base = __detail::get_ultimate_base(b);
    using matrix_type = decltype(a_base);
    using input_type = decltype(b_base);
    using output_type = std::remove_reference_t<decltype(c)>;
    using value_type = typename matrix_type::scalar_type;

    auto alpha_optional = __detail::get_scaling_factor(a, b);
    tensor_scalar_t<A> alpha = alpha_optional.value_or(1);
    
    value_type alpha_val = alpha;
    value_type beta = 0.0;
    // rocsparse_csr_set_pointers(matC, c.rowptr().data(), c.colind().data(), c.values().data());

    rocsparse_spgemm(handle_, rocsparse_operation_none,
                     rocsparse_operation_none, &alpha, matA, matB, &beta, matD,
                     matC, rocm_data_type<value_type>(),
                     rocsparse_spgemm_alg_default, rocsparse_spgemm_stage_numeric,
                     &buffer_size1_, workspace1_);
    hipDeviceSynchronize();
    // TODO: move the destory to the state not function
    // destroy matrix/vector descriptors
    // rocsparse_destroy_spmat_descr(matA);
    // rocsparse_destroy_spmat_descr(matB);
    // rocsparse_destroy_spmat_descr(matC);
    // rocsparse_destroy_spmat_descr(matD);
  }
  
private:
  rocsparse_handle handle_;
  std::shared_ptr<const allocator> alloc_;
  long unsigned int buffer_size1_;
  void* workspace1_;
  index<> result_shape_;
  index_t result_nnz_;
  rocsparse_spmat_descr matA, matB, matC, matD;
};

namespace __amdsparse {

struct operation_state_t {
  hipsparseSpGEMMDescr_t spgemm_descr;
};

} // namespace __amdsparse

} // namespace spblas
