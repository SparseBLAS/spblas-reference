#pragma once

#include <type_traits>

#include <cuda_runtime.h>
#include <cusparse.h>

#include <spblas/detail/operation_info_t.hpp>
#include <spblas/detail/ranges.hpp>
#include <spblas/detail/triangular_types.hpp>
#include <spblas/detail/view_inspectors.hpp>

#include "types.hpp"

namespace spblas {
class triangular_solve_handle_t {
public:
  triangular_solve_handle_t(std::shared_ptr<const allocator> alloc)
      : alloc_(alloc), buffer_size_(0) {
    cusparseCreate(&handle_);
    cusparseSpSV_createDescr(&spsv_desc_);
  }

  ~triangular_solve_handle_t() {
    alloc_->free(workspace_);
    cusparseSpSV_destroyDescr(spsv_desc_);
    cusparseDestroy(handle_);
  }

  template <matrix A, class Triangle, class DiagonalStorage, vector B, vector C>
    requires __detail::has_csr_base<A> &&
             __detail::has_contiguous_range_base<B> &&
             __ranges::contiguous_range<C>
  void triangular_solve_compute(A&& a, Triangle uplo, DiagonalStorage diag,
                                B&& b, C&& c) {
    auto a_base = __detail::get_ultimate_base(a);
    auto b_base = __detail::get_ultimate_base(b);
    using matrix_type = decltype(a_base);
    using input_type = decltype(b_base);
    using output_type = std::remove_reference_t<decltype(c)>;
    using value_type = typename matrix_type::scalar_type;
    // TODO: how to we provide the information with the matrix A
    auto diag_type = CUSPARSE_DIAG_TYPE_NON_UNIT;
    auto fill_mode = CUSPARSE_FILL_MODE_LOWER;
    value_type alpha = 1.0;
    cusparseSpMatDescr_t matA;
    cusparseDnVecDescr_t vecB, vecC;
    cusparseCreateDnVec(&vecB, b_base.size(), b_base.data(),
                        cuda_data_type<typename input_type::value_type>());
    cusparseCreateDnVec(&vecC, c.size(), c.data(),
                        cuda_data_type<typename output_type::value_type>());
    // Create sparse matrix A in CSR format
    cusparseCreateCsr(&matA, __backend::shape(a_base)[0],
                      __backend::shape(a_base)[1], a_base.values().size(),
                      a_base.rowptr().data(), a_base.colind().data(),
                      a_base.values().data(),
                      cusparse_index_type<typename matrix_type::offset_type>(),
                      cusparse_index_type<typename matrix_type::index_type>(),
                      CUSPARSE_INDEX_BASE_ZERO, cuda_data_type<value_type>());
    cusparseSpMatSetAttribute(matA, CUSPARSE_SPMAT_FILL_MODE, &fill_mode,
                              sizeof(fill_mode));
    cusparseSpMatSetAttribute(matA, CUSPARSE_SPMAT_DIAG_TYPE, &diag_type,
                              sizeof(diag_type));
    long unsigned int buffer_size = 0;
    cusparseSpSV_bufferSize(handle_, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha,
                            matA, vecB, vecC, cuda_data_type<value_type>(),
                            CUSPARSE_SPSV_ALG_DEFAULT, spsv_desc_,
                            &buffer_size);
    // only allocate the new workspace when the requiring workspace larger than
    // current
    if (buffer_size > buffer_size_) {
      buffer_size_ = buffer_size;
      alloc_->free(workspace_);
      alloc_->alloc(&workspace_, buffer_size);
    }
    cusparseSpSV_analysis(handle_, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha,
                          matA, vecB, vecC, cuda_data_type<value_type>(),
                          CUSPARSE_SPSV_ALG_DEFAULT, spsv_desc_, workspace_);
    cusparseDestroyDnVec(vecC);
    cusparseDestroyDnVec(vecB);
    cusparseDestroySpMat(matA);
  }

  template <matrix A, class Triangle, class DiagonalStorage, vector B, vector C>
    requires __detail::has_csr_base<A> &&
             __detail::has_contiguous_range_base<B> &&
             __ranges::contiguous_range<C>
  void triangular_solve_execute(A&& a, Triangle uplo, DiagonalStorage diag,
                                B&& b, C&& c) {
    auto a_base = __detail::get_ultimate_base(a);
    auto b_base = __detail::get_ultimate_base(b);
    using matrix_type = decltype(a_base);
    using input_type = decltype(b_base);
    using output_type = std::remove_reference_t<decltype(c)>;
    using value_type = typename matrix_type::scalar_type;
    auto alpha_optional = __detail::get_scaling_factor(a, b);
    tensor_scalar_t<A> alpha = alpha_optional.value_or(1);
    value_type alpha_val = alpha;
    cusparseSpMatDescr_t matA;
    cusparseDnVecDescr_t vecB, vecC;
    cusparseCreateDnVec(&vecB, b_base.size(), b_base.data(),
                        cuda_data_type<typename input_type::value_type>());
    cusparseCreateDnVec(&vecC, c.size(), c.data(),
                        cuda_data_type<typename output_type::value_type>());
    // Create sparse matrix A in CSR format
    cusparseCreateCsr(&matA, __backend::shape(a_base)[0],
                      __backend::shape(a_base)[1], a_base.values().size(),
                      a_base.rowptr().data(), a_base.colind().data(),
                      a_base.values().data(),
                      cusparse_index_type<typename matrix_type::offset_type>(),
                      cusparse_index_type<typename matrix_type::index_type>(),
                      CUSPARSE_INDEX_BASE_ZERO, cuda_data_type<value_type>());
    auto diag_type = std::is_same_v<DiagonalStorage, explicit_diagonal_t>
                         ? CUSPARSE_DIAG_TYPE_NON_UNIT
                         : CUSPARSE_DIAG_TYPE_UNIT;
    auto fill_mode = std::is_same_v<Triangle, upper_triangle_t>
                         ? CUSPARSE_FILL_MODE_UPPER
                         : CUSPARSE_FILL_MODE_LOWER;
    cusparseSpMatSetAttribute(matA, CUSPARSE_SPMAT_FILL_MODE, &fill_mode,
                              sizeof(fill_mode));
    cusparseSpMatSetAttribute(matA, CUSPARSE_SPMAT_DIAG_TYPE, &diag_type,
                              sizeof(diag_type));

    cusparseSpSV_solve(handle_, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha_val,
                       matA, vecB, vecC, cuda_data_type<value_type>(),
                       CUSPARSE_SPSV_ALG_DEFAULT, spsv_desc_);
    cusparseDestroyDnVec(vecC);
    cusparseDestroyDnVec(vecB);
    cusparseDestroySpMat(matA);
  }

private:
  cusparseHandle_t handle_;
  cusparseSpSVDescr_t spsv_desc_;
  std::shared_ptr<const allocator> alloc_;
  long unsigned int buffer_size_;
  void* workspace_;
};

template <matrix A, class Triangle, class DiagonalStorage, vector B, vector C>
  requires __detail::has_csr_base<A> &&
           __detail::has_contiguous_range_base<B> &&
           __ranges::contiguous_range<C>
void triangular_solve_inspect(triangular_solve_handle_t& trisolve_handle, A&& a,
                              Triangle uplo, DiagonalStorage diag, B&& b,
                              C&& c) {}

template <matrix A, class Triangle, class DiagonalStorage, vector B, vector C>
  requires __detail::has_csr_base<A> &&
           __detail::has_contiguous_range_base<B> &&
           __ranges::contiguous_range<C>
void triangular_solve_compute(triangular_solve_handle_t& trisolve_handle, A&& a,
                              Triangle uplo, DiagonalStorage diag, B&& b,
                              C&& c) {
  trisolve_handle.triangular_solve_compute(a, uplo, diag, b, c);
}

template <matrix A, class Triangle, class DiagonalStorage, vector B, vector C>
  requires __detail::has_csr_base<A> &&
           __detail::has_contiguous_range_base<B> &&
           __ranges::contiguous_range<C>
void triangular_solve_execute(triangular_solve_handle_t& trisolve_handle, A&& a,
                              Triangle uplo, DiagonalStorage diag, B&& b,
                              C&& c) {
  trisolve_handle.triangular_solve_execute(a, uplo, diag, b, c);
}

} // namespace spblas
