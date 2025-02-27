#pragma once

#include <type_traits>

#include <hip/hip_runtime.h>
#include <hipsparse/hipsparse.h>

#include <spblas/detail/operation_info_t.hpp>
#include <spblas/detail/ranges.hpp>
#include <spblas/detail/view_inspectors.hpp>

#include "types.hpp"

namespace spblas {
class conversion_handle_t {
public:
  conversion_handle_t(std::shared_ptr<const allocator> alloc)
      : alloc_(alloc), buffer_size_(0) {
    hipsparseCreate(&handle_);
  }

  ~conversion_handle_t() {
    alloc_->free(workspace_);
    hipsparseDestroy(handle_);
  }

  auto result_nnz() {
    return result_nnz_;
  }

  template <matrix A, matrix B>
    requires __detail::has_mdspan_matrix_base<A> && __detail::has_csr_base<B>
  void conversion_compute(A&& a, B&& b) {
    auto a_base = __detail::get_ultimate_base(a);
    auto b_base = __detail::get_ultimate_base(b);
    using matrix_type = decltype(a_base);
    using output_type = decltype(b_base);
    using value_type = typename output_type::scalar_type;
    hipsparseDnMatDescr_t matA;
    hipsparseSpMatDescr_t matB;
    hipsparseCreateDnMat(&matA, a_base.extent(0), a_base.extent(1),
                        a_base.extent(1), a_base.data_handle(),
                        hip_data_type<typename matrix_type::value_type>(),
                        HIPSPARSE_ORDER_ROW);
    // Create sparse matrix C in CSR format
    hipsparseCreateCsr(&matB, __backend::shape(a_base)[0],
                      __backend::shape(a_base)[1], b_base.values().size(),
                      b_base.rowptr().data(), b_base.colind().data(),
                      b_base.values().data(),
                      hipsparse_index_type<typename output_type::offset_type>(),
                      hipsparse_index_type<typename output_type::index_type>(),
                      HIPSPARSE_INDEX_BASE_ZERO, hip_data_type<value_type>());
    long unsigned int buffer_size = 0;
    hipsparseDenseToSparse_bufferSize(
        handle_, matA, matB, HIPSPARSE_DENSETOSPARSE_ALG_DEFAULT, &buffer_size);
    // only allocate the new workspace when the requiring workspace larger than
    // current
    if (buffer_size > buffer_size_) {
      buffer_size_ = buffer_size;
      alloc_->free(workspace_);
      alloc_->alloc(&workspace_, buffer_size);
    }
    hipsparseDenseToSparse_analysis(
        handle_, matA, matB, HIPSPARSE_DENSETOSPARSE_ALG_DEFAULT, workspace_);
    int64_t num_rows, num_cols;
    hipsparseSpMatGetSize(matB, &num_rows, &num_cols, &result_nnz_);
    hipsparseDestroyDnMat(matA);
    hipsparseDestroySpMat(matB);
  }

  template <matrix A, matrix B>
    requires __detail::has_mdspan_matrix_base<A> && __detail::has_csr_base<B>
  void conversion_execute(A&& a, B&& b) {
    auto a_base = __detail::get_ultimate_base(a);
    auto b_base = __detail::get_ultimate_base(b);
    using matrix_type = decltype(a_base);
    using output_type = decltype(b_base);
    using value_type = typename output_type::scalar_type;
    hipsparseDnMatDescr_t matA;
    hipsparseSpMatDescr_t matB;
    hipsparseCreateDnMat(&matA, a_base.extent(0), a_base.extent(1),
                        a_base.extent(1), a_base.data_handle(),
                        hip_data_type<typename matrix_type::value_type>(),
                        HIPSPARSE_ORDER_ROW);
    // Create sparse matrix C in CSR format
    hipsparseCreateCsr(&matB, __backend::shape(a_base)[0],
                      __backend::shape(a_base)[1], b_base.values().size(),
                      b_base.rowptr().data(), b_base.colind().data(),
                      b_base.values().data(),
                      hipsparse_index_type<typename output_type::offset_type>(),
                      hipsparse_index_type<typename output_type::index_type>(),
                      HIPSPARSE_INDEX_BASE_ZERO, hip_data_type<value_type>());
    hipsparseDenseToSparse_convert(
        handle_, matA, matB, HIPSPARSE_DENSETOSPARSE_ALG_DEFAULT, workspace_);
  }

private:
  hipsparseHandle_t handle_;
  std::shared_ptr<const allocator> alloc_;
  long unsigned int buffer_size_;
  void* workspace_;
  index_t result_nnz_;
};

template <matrix A, matrix B>
  requires __detail::has_mdspan_matrix_base<A> && __detail::has_csr_base<B>
void conversion_inspect(conversion_handle_t& handle, A&& a, B&& b) {}

template <matrix A, matrix B>
  requires __detail::has_mdspan_matrix_base<A> && __detail::has_csr_base<B>
void conversion_compute(conversion_handle_t& handle, A&& a, B&& b) {
  handle.conversion_compute(a, b);
}

template <matrix A, matrix B>
  requires __detail::has_mdspan_matrix_base<A> && __detail::has_csr_base<B>
void conversion_execute(conversion_handle_t& handle, A&& a, B&& b) {
  handle.conversion_execute(a, b);
}

} // namespace spblas
