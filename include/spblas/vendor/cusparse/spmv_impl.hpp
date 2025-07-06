#pragma once

#include <cusparse.h>

#include <spblas/detail/log.hpp>
#include <spblas/detail/operation_info_t.hpp>
#include <spblas/detail/ranges.hpp>
#include <spblas/detail/view_inspectors.hpp>
#include <spblas/vendor/cusparse/detail/cusparse_tensors.hpp>
#include <spblas/vendor/cusparse/detail/get_transpose.hpp>
#include <spblas/vendor/cusparse/operation_state_t.hpp>
#include <spblas/vendor/cusparse/types.hpp>

namespace spblas {

template <matrix A, vector X, vector Y>
  requires(__detail::has_csr_base<A> &&
           __detail::has_contiguous_range_base<X> &&
           __ranges::contiguous_range<Y> &&
           detail::has_valid_cusparse_types_v<A> &&
           detail::has_valid_cusparse_types_v<X> &&
           detail::has_valid_cusparse_types_v<Y>)
void multiply(operation_info_t& info, A&& a, X&& x, Y&& y) {
  log_trace("");

  auto x_base = __detail::get_ultimate_base(x);
  auto a_base = __detail::get_ultimate_base(a);

  auto alpha_optional = __detail::get_scaling_factor(a, x);
  tensor_scalar_t<A> alpha = alpha_optional.value_or(1);
  tensor_scalar_t<A> beta = 0;

  // Get or create state
  auto state = info.state_.get_state<__cusparse::spmv_state_t>();
  if (!state) {
    info.state_ = __cusparse::operation_state_t(
        std::make_unique<__cusparse::spmv_state_t>());
    state = info.state_.get_state<__cusparse::spmv_state_t>();
  }

  // Create or get matrix descriptor
  if (!state->a_descriptor()) {
    cusparseSpMatDescr_t a_descr = __cusparse::create_cusparse_handle(a_base);
    state->set_a_descriptor(a_descr);
  }

  // Create vector descriptors
  cusparseDnVecDescr_t b_descr = __cusparse::create_cusparse_handle(x_base);
  cusparseDnVecDescr_t c_descr = __cusparse::create_cusparse_handle(y);
  state->set_b_descriptor(b_descr);
  state->set_c_descriptor(c_descr);

  // Get operation type based on matrix format
  auto a_transpose = __cusparse::get_transpose(a);

  // Get buffer size
  size_t buffer_size;
  __cusparse::throw_if_error(cusparseSpMV_bufferSize(
      state->handle(), a_transpose, &alpha, state->a_descriptor(),
      state->b_descriptor(), &beta, state->c_descriptor(),
      detail::cuda_data_type_v<tensor_scalar_t<Y>>, CUSPARSE_SPMV_ALG_DEFAULT,
      &buffer_size));

  // Allocate buffer if needed
  void* buffer = nullptr;
  if (buffer_size > 0) {
    cudaMalloc(&buffer, buffer_size);
  }

  // Execute SpMV
  __cusparse::throw_if_error(
      cusparseSpMV(state->handle(), a_transpose, &alpha, state->a_descriptor(),
                   state->b_descriptor(), &beta, state->c_descriptor(),
                   detail::cuda_data_type_v<tensor_scalar_t<Y>>,
                   CUSPARSE_SPMV_ALG_DEFAULT, buffer));

  // Free buffer if allocated
  if (buffer) {
    cudaFree(buffer);
  }
}

template <matrix A, vector X, vector Y>
  requires(__detail::has_csr_base<A> &&
           __detail::has_contiguous_range_base<X> &&
           __ranges::contiguous_range<Y> &&
           detail::has_valid_cusparse_types_v<A> &&
           detail::has_valid_cusparse_types_v<X> &&
           detail::has_valid_cusparse_types_v<Y>)
void multiply(A&& a, X&& x, Y&& y) {
  operation_info_t info;
  multiply(info, std::forward<A>(a), std::forward<X>(x), std::forward<Y>(y));
}

} // namespace spblas
