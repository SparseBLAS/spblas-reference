#pragma once
#include "operation_state_t.hpp"
#include "trisolve.hpp"
#include <spblas/detail/operation_info_t.hpp>

namespace spblas {

template <matrix A, class Triangle, class DiagonalStorage, vector B, vector C>
  requires __detail::has_csr_base<A> &&
           __detail::has_contiguous_range_base<B> &&
           __ranges::contiguous_range<C>
void triangular_solve(operation_info_t& info, A&& a, Triangle uplo,
                      DiagonalStorage diag, B&& b, C&& c) {
  // Get or create state
  auto state = info.state_.get_state<triangular_solve_state_t>();
  if (!state) {
    info.state_ = __rocsparse::operation_state_t(
        std::make_unique<triangular_solve_state_t>());
    state = info.state_.get_state<triangular_solve_state_t>();
  }
  state->triangular_solve(a, uplo, diag, b, c);
}

} // namespace spblas
