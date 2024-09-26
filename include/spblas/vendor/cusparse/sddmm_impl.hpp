#pragma once

#include <type_traits>

#include <cuda_runtime.h>
#include <cusparse.h>

#include <spblas/detail/operation_info_t.hpp>
#include <spblas/detail/ranges.hpp>
#include <spblas/detail/view_inspectors.hpp>

#include "types.hpp"

namespace spblas {
class sampled_multiply_handle_t {
public:
  sampled_multiply_handle_t(std::shared_ptr<const allocator> alloc)
      : alloc_(alloc), buffer_size_(0) {
    cusparseCreate(&handle_);
  }

  ~sampled_multiply_handle_t() {
    alloc_->free(workspace_);
    cusparseDestroy(handle_);
  }

  template <matrix A, matrix B, matrix C>
    requires __detail::has_mdspan_matrix_base<A> &&
             __detail::has_mdspan_matrix_base<B> && __detail::has_csr_base<C>
  void sampled_multiply_compute(A&& a, B&& b, C&& c) {
    auto a_base = __detail::get_ultimate_base(a);
    auto b_base = __detail::get_ultimate_base(b);
    auto c_base = __detail::get_ultimate_base(c);
    using matrix_type = decltype(a_base);
    using input_type = decltype(b_base);
    using output_type = std::remove_reference_t<decltype(c)>;
    using value_type = typename output_type::scalar_type;
    value_type alpha = 1.0;
    value_type beta = 0.0;
    cusparseDnMatDescr_t matA, matB;
    cusparseSpMatDescr_t matC;
    cusparseCreateDnMat(&matA, a_base.extent(0), a_base.extent(1),
                        a_base.extent(1), a_base.data_handle(),
                        cuda_data_type<typename matrix_type::value_type>(),
                        CUSPARSE_ORDER_ROW);
    cusparseCreateDnMat(&matB, b_base.extent(0), b_base.extent(1),
                        b_base.extent(1), b_base.data_handle(),
                        cuda_data_type<typename input_type::value_type>(),
                        CUSPARSE_ORDER_ROW);
    // Create sparse matrix C in CSR format
    cusparseCreateCsr(&matC, __backend::shape(c_base)[0],
                      __backend::shape(c_base)[1], c_base.values().size(),
                      c_base.rowptr().data(), c_base.colind().data(),
                      c_base.values().data(),
                      cusparse_index_type<typename output_type::offset_type>(),
                      cusparse_index_type<typename output_type::index_type>(),
                      CUSPARSE_INDEX_BASE_ZERO, cuda_data_type<value_type>());
    long unsigned int buffer_size = 0;
    cusparseSDDMM_bufferSize(handle_, CUSPARSE_OPERATION_NON_TRANSPOSE,
                             CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA,
                             matB, &beta, matC, cuda_data_type<value_type>(),
                             CUSPARSE_SDDMM_ALG_DEFAULT, &buffer_size);
    // only allocate the new workspace when the requiring workspace larger than
    // current
    if (buffer_size > buffer_size_) {
      buffer_size_ = buffer_size;
      alloc_->free(workspace_);
      alloc_->alloc(&workspace_, buffer_size);
    }
    cusparseSDDMM_preprocess(handle_, CUSPARSE_OPERATION_NON_TRANSPOSE,
                             CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA,
                             matB, &beta, matC, cuda_data_type<value_type>(),
                             CUSPARSE_SDDMM_ALG_DEFAULT, workspace_);
    cusparseDestroySpMat(matC);
    cusparseDestroyDnMat(matB);
    cusparseDestroyDnMat(matA);
  }

  template <matrix A, matrix B, matrix C>
    requires __detail::has_mdspan_matrix_base<A> &&
             __detail::has_mdspan_matrix_base<B> && __detail::has_csr_base<C>
  void sampled_multiply_execute(A&& a, B&& b, C&& c) {
    auto a_base = __detail::get_ultimate_base(a);
    auto b_base = __detail::get_ultimate_base(b);
    auto c_base = __detail::get_ultimate_base(c);
    using matrix_type = decltype(a_base);
    using input_type = decltype(b_base);
    using output_type = std::remove_reference_t<decltype(c)>;
    using value_type = typename output_type::scalar_type;
    auto alpha_optional = __detail::get_scaling_factor(a, b);
    tensor_scalar_t<A> alpha = alpha_optional.value_or(1);
    value_type alpha_val = alpha;
    value_type beta = 0.0;
    cusparseDnMatDescr_t matA, matB;
    cusparseSpMatDescr_t matC;
    cusparseCreateDnMat(&matA, a_base.extent(0), a_base.extent(1),
                        a_base.extent(1), a_base.data_handle(),
                        cuda_data_type<typename matrix_type::value_type>(),
                        CUSPARSE_ORDER_ROW);
    cusparseCreateDnMat(&matB, b_base.extent(0), b_base.extent(1),
                        b_base.extent(1), b_base.data_handle(),
                        cuda_data_type<typename input_type::value_type>(),
                        CUSPARSE_ORDER_ROW);
    // Create sparse matrix C in CSR format
    cusparseCreateCsr(&matC, __backend::shape(c_base)[0],
                      __backend::shape(c_base)[1], c_base.values().size(),
                      c_base.rowptr().data(), c_base.colind().data(),
                      c_base.values().data(),
                      cusparse_index_type<typename output_type::offset_type>(),
                      cusparse_index_type<typename output_type::index_type>(),
                      CUSPARSE_INDEX_BASE_ZERO, cuda_data_type<value_type>());
    cusparseSDDMM(handle_, CUSPARSE_OPERATION_NON_TRANSPOSE,
                  CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha_val, matA, matB,
                  &beta, matC, cuda_data_type<value_type>(),
                  CUSPARSE_SDDMM_ALG_DEFAULT, workspace_);
    cusparseDestroyDnMat(matA);
    cusparseDestroyDnMat(matB);
    cusparseDestroySpMat(matC);
  }

private:
  cusparseHandle_t handle_;
  std::shared_ptr<const allocator> alloc_;
  long unsigned int buffer_size_;
  void* workspace_;
};

template <matrix A, matrix B, matrix C>
  requires __detail::has_mdspan_matrix_base<A> &&
           __detail::has_mdspan_matrix_base<B> && __detail::has_csr_base<C>
void sampled_multiply_inspect(sampled_multiply_handle_t& sddmm_handle, A&& a,
                              B&& b, C&& c) {}

template <matrix A, matrix B, matrix C>
  requires __detail::has_mdspan_matrix_base<A> &&
           __detail::has_mdspan_matrix_base<B> && __detail::has_csr_base<C>
void sampled_multiply_compute(sampled_multiply_handle_t& sddmm_handle, A&& a,
                              B&& b, C&& c) {
  sddmm_handle.sampled_multiply_compute(a, b, c);
}

template <matrix A, matrix B, matrix C>
  requires __detail::has_mdspan_matrix_base<A> &&
           __detail::has_mdspan_matrix_base<B> && __detail::has_csr_base<C>
void sampled_multiply_execute(sampled_multiply_handle_t& sddmm_handle, A&& a,
                              B&& b, C&& c) {
  sddmm_handle.sampled_multiply_execute(a, b, c);
}

} // namespace spblas
