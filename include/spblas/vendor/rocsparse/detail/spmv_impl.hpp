#pragma once

#include <rocsparse/rocsparse.h>

#include <spblas/detail/operation_info_t.hpp>
#include <spblas/detail/ranges.hpp>
#include <spblas/detail/view_inspectors.hpp>
#include <spblas/vendor/rocsparse/detail/get_transpose.hpp>
#include <spblas/vendor/rocsparse/detail/rocsparse_tensors.hpp>
#include <spblas/vendor/rocsparse/detail/spmv_state_t.hpp>
#include <spblas/vendor/rocsparse/operation_state_t.hpp>
#include <spblas/vendor/rocsparse/type_validation.hpp>

namespace spblas {

template <matrix A, vector B, vector C>
  requires(__detail::has_csr_base<A> &&
           __detail::has_contiguous_range_base<B> &&
           __ranges::contiguous_range<C> &&
           detail::has_valid_rocsparse_matrix_types_v<A> &&
           detail::has_valid_rocsparse_vector_types_v<B> &&
           detail::has_valid_rocsparse_vector_types_v<C>)
void multiply(operation_info_t& info, A&& a, B&& b, C&& c) {
  auto a_base = __detail::get_ultimate_base(a);
  auto b_base = __detail::get_ultimate_base(b);

  auto alpha_optional = __detail::get_scaling_factor(a, b);
  tensor_scalar_t<A> alpha = alpha_optional.value_or(1);
  tensor_scalar_t<A> beta = 0;

  // Get or create state
  auto state = info.state_.get_state<__rocsparse::spmv_state_t>();
  if (!state) {
    info.state_ = __rocsparse::operation_state_t(
        std::make_unique<__rocsparse::spmv_state_t>());
    state = info.state_.get_state<__rocsparse::spmv_state_t>();
  }

  // Create descriptors
  auto a_descr = __rocsparse::create_rocsparse_handle(a_base);
  auto b_descr = __rocsparse::create_rocsparse_handle(b_base);
  auto c_descr = __rocsparse::create_rocsparse_handle(c);

  state->set_a_descriptor(a_descr);
  state->set_b_descriptor(b_descr);
  state->set_c_descriptor(c_descr);

  // Get operation type based on matrix format
  auto a_transpose = __rocsparse::get_transpose(a);

  // Get buffer size
  size_t buffer_size = 0;
  __rocsparse::throw_if_error(rocsparse_spmv(
      state->handle(), a_transpose, &alpha, state->a_descriptor(),
      state->b_descriptor(), &beta, state->c_descriptor(),
      detail::rocsparse_data_type_v<tensor_scalar_t<A>>,
      rocsparse_spmv_alg_csr_stream, rocsparse_spmv_stage_buffer_size,
      &buffer_size, nullptr));

  // Allocate buffer if needed
  state->allocate_workspace(buffer_size);

  // Execute SpMV
  __rocsparse::throw_if_error(rocsparse_spmv(
      state->handle(), a_transpose, &alpha, state->a_descriptor(),
      state->b_descriptor(), &beta, state->c_descriptor(),
      detail::rocsparse_data_type_v<tensor_scalar_t<A>>,
      rocsparse_spmv_alg_csr_stream, rocsparse_spmv_stage_compute, &buffer_size,
      state->workspace()));
}

template <matrix A, vector B, vector C>
  requires(__detail::has_csr_base<A> &&
           __detail::has_contiguous_range_base<B> &&
           __ranges::contiguous_range<C> &&
           detail::has_valid_rocsparse_matrix_types_v<A> &&
           detail::has_valid_rocsparse_vector_types_v<B> &&
           detail::has_valid_rocsparse_vector_types_v<C>)
void multiply(A&& a, B&& b, C&& c) {
  operation_info_t info;
  multiply(info, std::forward<A>(a), std::forward<B>(b), std::forward<C>(c));
}

} // namespace spblas
