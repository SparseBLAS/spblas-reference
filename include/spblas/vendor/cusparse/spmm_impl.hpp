#pragma once

#include <cusparse.h>

#include <stdexcept>

#include <spblas/detail/log.hpp>
#include <spblas/detail/operation_info_t.hpp>
#include <spblas/detail/ranges.hpp>
#include <spblas/detail/view_inspectors.hpp>
#include <spblas/vendor/cusparse/detail/cusparse_tensors.hpp>
#include <spblas/vendor/cusparse/detail/get_transpose.hpp>
#include <spblas/vendor/cusparse/detail/spmm_state_t.hpp>
#include <spblas/vendor/cusparse/operation_state_t.hpp>
#include <spblas/vendor/cusparse/type_validation.hpp>

#include <spblas/vendor/cusparse/detail/detail.hpp>

//
// Defines the following APIs for SpMM:
//
//  Y = alpha * op(A) * X
//
//  where A is a sparse matrices of CSR format and
//  X/Y are dense matrices of row_major format
//
// //operation_info_t multiply_inspect(A, x, y)
// //void multiply_inspect(operation_info_t, A, x, y)
//
// //void multiply_compute(operation_info_t, A, x, y)
// void multiply(A, x, y)
//

namespace spblas {

template <matrix A, matrix X, matrix Y>
  requires(
      (__detail::has_csr_base<A> || __detail::has_csc_base<A>) &&
      __detail::has_mdspan_matrix_base<X> && __detail::is_matrix_mdspan_v<Y> &&
      std::is_same_v<typename __detail::ultimate_base_type_t<X>::layout_type,
                     __mdspan::layout_right> &&
      std::is_same_v<typename std::remove_cvref_t<Y>::layout_type,
                     __mdspan::layout_right>)
void multiply(ExecutionPolicy&& policy, A&& a, X&& x, Y&& y) {
  log_trace("");

  auto x_base = __detail::get_ultimate_base(x);
  auto y_base = __detail::get_ultimate_base(y);

  if (__detail::is_conjugated(x) || __detail::is_conjugated(y)) {
    throw std::runtime_error(
        "cusparse backend does not support conjugated dense matrices.");
  }

  auto alpha_optional = __detail::get_scaling_factor(a, x);
  tensor_scalar_t<A> alpha = alpha_optional.value_or(1);
  tensor_scalar_t<A> beta = 0;

  // Get or create state
  auto state = info.state_.get_state<__cusparse::spmm_state_t>();
  if (!state) {
    info.state_ = __cusparse::operation_state_t(
        std::make_unique<__cusparse::spmm_state_t>());
    state = info.state_.get_state<__cusparse::spmm_state_t>();
  }

  auto a_handle = __cusparse::get_matrix_handle(a);
  auto a_transpose = __cusparse::get_transpose(a);

  cusparseDnMatDescr_t x_handle, y_handle;

  __cusparse::throw_if_error(cusparseCreateDnMat(&x_handle, x_base.extent(0),
      x_base.extent(1), x_base.extent(0), x_base.data(),
      detail::cuda_data_type_v<tensor_scalar_t<X>>, CUSPARSE_ORDER_ROW);

  __cusparse::throw_if_error(cusparseCreateDnMat(&y_handle, y_base.extent(0),
      y_base.extent(1), y_base.extent(0), y_base.data(),
      detail::cuda_data_type_v<tensor_scalar_t<Y>>, CUSPARSE_ORDER_ROW);

  // Get buffer size
  size_t buffer_size;
  __cusparse::throw_if_error(cusparseSpMM_bufferSize(
      state->handle(), a_transpose, b_transpose, &alpha, a_handle,
      x_handle, &beta, y_handle, detail::cuda_data_type_v<tensor_scalar_t<Y>>,
      CUSPARSE_SPMM_ALG_DEFAULT, &buffer_size));

  // Allocate buffer if needed
  void* buffer = nullptr;
  if (buffer_size > 0) {
    cudaMalloc(&buffer, buffer_size);
  }

  // Execute SpMM
  __cusparse::throw_if_error(
      cusparseSpMV(state->handle(), a_transpose, &alpha, a_handle,
                   x_handle, &beta, y_handle,
                   detail::cuda_data_type_v<tensor_scalar_t<Y>>,
                   CUSPARSE_SPMM_ALG_DEFAULT, buffer));

  // Free buffer if allocated
  if (buffer) {
    cudaFree(buffer);
  }
}

template <matrix A, vector X, vector Y>
  requires(__detail::has_csr_base<A> &&
           __detail::has_contiguous_range_base<X> &&
           __ranges::contiguous_range<Y> &&
           detail::has_valid_cusparse_matrix_types_v<A> &&
           detail::has_valid_cusparse_vector_types_v<X> &&
           detail::has_valid_cusparse_vector_types_v<Y>)
void multiply(A&& a, X&& x, Y&& y) {
  operation_info_t info;
  multiply(info, std::forward<A>(a), std::forward<X>(x), std::forward<Y>(y));
}

} // namespace spblas
