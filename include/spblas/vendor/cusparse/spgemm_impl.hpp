#pragma once

#include <cusparse.h>

#include <stdexcept>

#include <spblas/detail/log.hpp>

#include <spblas/algorithms/transposed.hpp>
#include <spblas/detail/operation_info_t.hpp>
#include <spblas/detail/ranges.hpp>
#include <spblas/detail/view_inspectors.hpp>
#include <spblas/views/matrix_opt.hpp>

#include <spblas/vendor/cusparse/detail/detail.hpp>
#include <spblas/vendor/cusparse/detail/get_transpose.hpp>
#include <spblas/vendor/cusparse/detail/spgemm_state_t.hpp>
#include <spblas/vendor/cusparse/exception.hpp>
#include <spblas/vendor/cusparse/operation_state_t.hpp>
#include <spblas/vendor/cusparse/type_validation.hpp>
#include <spblas/vendor/cusparse/types.hpp>

//
// Defines the following APIs for SpGEMM:
//
//  C = op(A) * op(B)
//
//  where A,B and C are sparse matrices of CSR format
//
// operation_info_t multiply_inspect(A, B, C)
// void multiply_compute(operation_info_t, A, B, C)
//

namespace spblas {

template <matrix A, matrix B, matrix C>
  requires(__detail::has_csr_base<A> || __detail::has_csc_base<A>) &&
          (__detail::has_csr_base<B> || __detail::has_csc_base<B>) &&
          __detail::is_csr_view_v<C>
operation_info_t
    multiply_compute(ExecutionPolicy&& policy, A&& a, B&& b, C&& c) {
  log_trace("");

  if (__detail::is_conjugated(c)) {
    throw std::runtime_error(
        "cusparse backend does not support conjugated output matrices.");
  }

  // Get or create state
  auto state = info.state_.get_state<__cusparse::spgemm_state_t>();
  if (!state) {
    info.state_ = __cusparse::operation_state_t(
        std::make_unique<__cusparse::spgemm_state_t>());
    state = info.state_.get_state<__cusparse::spgemm_state_t>();
  }

  auto handle = state->handle();

  // Create or get matrix descriptors
  auto a_handle = __cusparse::get_matrix_handle(a);
  auto b_handle = __cusparse::get_matrix_handle(b);
  auto c_handle = __cusparse::get_matrix_handle(c);

  // Get operation type based on matrix format
  auto a_transpose = __cusparse::get_transpose(a);
  auto b_transpose = __cusparse::get_transpose(b);

  auto alpha_optional = __detail::get_scaling_factor(a, x);
  tensor_scalar_t<A> alpha = alpha_optional.value_or(1);
  tensor_scalar_t<A> beta = 0;

  using T = tensor_scalar_t<C>;
  using I = tensor_index_t<C>;
  using O = tensor_offset_t<C>;

  O* c_rowptr;
  if (c.rowptr().size() >= __backend::shape(c)[0] + 1) {
    c_rowptr = c.rowptr().data();
  } else {
    cudaMalloc(&c_rowptr, (__backend::shape(c)[0] + 1) * sizeof(O));
  }

  // Create SpGEMM descriptor
  cusparseSpGEMMDescr_t spgemm_descr;
  __cusparse::throw_if_error(cusparseSpGEMM_createDescr(&spgemm_descr));

  // Work estimation (get buffer size)
  size_t bufferSize1 = 0;
  __cusparse::throw_if_error(cusparseSpGEMM_workEstimation(
      handle, a_transpose, b_transpose, &alpha, a_handle, b_handle,
      &beta, c_handle, detail::cuda_data_type_v<T>, CUSPARSE_SPGEMM_DEFAULT,
      spgemm_descr, &bufferSize1, nullptr));

  void* buffer1 = nullptr;
  if (bufferSize1 > 0) {
    cudaMalloc(&buffer1, bufferSize1);
  }

  // Work estimation (execute)
  __cusparse::throw_if_error(cusparseSpGEMM_workEstimation(
      handle, a_transpose, b_transpose, &alpha, a_handle, b_handle,
       &beta, c_handle, detail::cuda_data_type_v<T>, CUSPARSE_SPGEMM_DEFAULT,
      spgemm_descr, &bufferSize1, buffer1));

  // Compute (get buffer size)
  size_t bufferSize2 = 0;
  __cusparse::throw_if_error(cusparseSpGEMM_compute(
      handle, a_transpose, b_transpose, &alpha, a_handle, b_handle,
      &beta, c_handle, detail::cuda_data_type_v<T>, CUSPARSE_SPGEMM_DEFAULT,
      spgemm_descr, &bufferSize2, nullptr));

  void* buffer2 = nullptr;
  if (bufferSize2 > 0) {
    cudaMalloc(&buffer2, bufferSize2);
  }

  // Compute (execute)
  __cusparse::throw_if_error(cusparseSpGEMM_compute(
      handle, a_transpose, b_transpose, &alpha, a_handle, b_handle,
       &beta, c_handle, detail::cuda_data_type_v<T>, CUSPARSE_SPGEMM_DEFAULT,
      spgemm_descr, &bufferSize2, buffer2));

  // Get output nnz
  size_t C_rows, C_cols, C_nnz;
  __cusparse::throw_if_error(
      cusparseSpMatGetSize(c_handle, &C_rows, &C_cols, &C_nnz));

  log_info("computed c_nnz = %ld", C_nnz);

  return operation_info_t{
      index<>{__backend::shape(c)[0], __backend::shape(c)[1]}, nnz,
      __mkl::operation_state_t{__detail::has_matrix_opt(a) ? nullptr : a_handle,
                               __detail::has_matrix_opt(b) ? nullptr : b_handle,
                               c_handle, nullptr, descr, (void*) c_rowptr, q}};
}

template <matrix A, matrix B, matrix C>
  requires(__detail::has_csr_base<A> || __detail::has_csc_base<A>) &&
          (__detail::has_csr_base<B> || __detail::has_csc_base<B>) &&
          __detail::is_csr_view_v<C>
void multiply_fill(ExecutionPolicy&& policy, operation_info_t& info, A&& a,
                   B&& b, C&& c) {
  log_trace("");

  if (__detail::is_conjugated(c)) {
    throw std::runtime_error(
        "cusparse backend does not support conjugated output matrices.");
  }

  using T = tensor_scalar_t<C>;
  using O = tensor_offset_t<C>;

  auto state = info.state_.get_state<__cusparse::spgemm_state_t>();

  auto handle = state->handle();

  // Get matrix descriptors
  auto a_handle = __cusparse::get_matrix_handle(a);
  auto b_handle = __cusparse::get_matrix_handle(b);
  auto c_handle = __cusparse::get_matrix_handle(c);
  auto spgemm_descr = state->spgemm_descriptor();

  // Get operation type based on matrix format
  auto a_transpose = __cusparse::get_transpose(a);
  auto b_transpose = __cusparse::get_transpose(b);

  auto alpha_optional = __detail::get_scaling_factor(a, b);
  tensor_scalar_t<A> alpha = alpha_optional.value_or(1);
  T beta = 0;

  // Update C descriptor with the now-allocated colind and values
  O* c_rowptr = static_cast<O*>(state->c_rowptr());
  __cusparse::throw_if_error(cusparseCsrSetPointers(
      c_handle, c_rowptr, c.colind().data(), c.values().data()));

  // Copy computed results into C's arrays
  __cusparse::throw_if_error(cusparseSpGEMM_copy(
      handle, a_transpose, b_transpose,
      &alpha, a_handle, b_handle, &beta, c_handle,
      detail::cuda_data_type_v<T>, CUSPARSE_SPGEMM_DEFAULT, spgemm_descr));

  if (c_rowptr != c.rowptr().data()) {
    cudaMemcpy(c.rowptr().data(), c_rowptr,
             sizeof(O) * (__backend::shape(c)[0] + 1))
        .wait();
  }

  if (alpha_optional.has_value()) {
    scale(alpha, c);
  }
}

template <matrix A, matrix B, matrix C>
  requires(__detail::has_csr_base<A> || __detail::has_csc_base<A>) &&
          (__detail::has_csr_base<B> || __detail::has_csc_base<B>) &&
          __detail::is_csc_view_v<C>
operation_info_t multiply_compute(A&& a, B&& b, C&& c) {
  return multiply_compute(transposed(b), transposed(a), transposed(c));
}

template <matrix A, matrix B, matrix C>
  requires((__detail::has_csr_base<A> || __detail::has_csc_base<A>) &&
           (__detail::has_csr_base<B> || __detail::has_csc_base<B>) &&
           __detail::is_csc_view_v<C>)
void multiply_fill(operation_info_t& info, A&& a, B&& b, C&& c) {
  multiply_fill(info, transposed(b), transposed(a), transposed(c));
}

} // namespace spblas
