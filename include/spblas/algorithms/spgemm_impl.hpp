#pragma once

#include <spblas/backend/backend.hpp>
#include <spblas/concepts.hpp>
#include <spblas/detail/api_concepts.hpp>
#include <spblas/detail/log.hpp>
#include <spblas/detail/operation_state.hpp>
#include <spblas/views/inspectors.hpp>

#include "detail/spgemm/spgemm.hpp"

//
// Reference SpGEMM: C = alpha * A * B
//
//   multiply_inspect([policy,] state, alpha, A, B, C)
//   multiply_compute([policy,] state, alpha, A, B, C)
//     Sets state.get_result_nnz(); the user then attaches arrays of that
//     size to C (rowptr/colptr sized by C's shape).
//   multiply_fill([policy,] state, alpha, A, B, C)
//
//   The supported A / B / C format combinations and their kernels are listed
//   in detail/spgemm/spgemm.hpp. C's pattern is symbolic: entries that are
//   numerically zero (e.g. alpha == 0) are kept.
//
//   Not yet supported: the [beta, D] term and the 3-stage
//   _compute_symbolic / _fill_symbolic / _numeric API.
//
// The policy-first overloads are in multiply_impl.hpp.
//

namespace spblas {

// The reference implementation has nothing to prepare.
template <__detail::scalar Alpha, matrix A, matrix B, matrix C>
  requires(__detail::is_csr_wrapper_v<C> || __detail::is_csc_wrapper_v<C>)
void multiply_inspect(operation_state& state, Alpha alpha, A&& a, B&& b,
                      C&& c) {
  log_trace("");
}

template <__detail::scalar Alpha, matrix A, matrix B, matrix C>
  requires((__detail::is_csr_wrapper_v<C> || __detail::is_csc_wrapper_v<C>) &&
           requires(A&& a, B&& b, C&& c) { __detail::spgemm_nnz(a, b, c); })
void multiply_compute(operation_state& state, Alpha alpha, A&& a, B&& b,
                      C&& c) {
  log_trace("");
  state.update_impl_(__detail::spgemm_nnz(a, b, c));
}

template <__detail::scalar Alpha, matrix A, matrix B, matrix C>
  requires((__detail::is_csr_wrapper_v<C> || __detail::is_csc_wrapper_v<C>) &&
           requires(A&& a, B&& b, C&& c) { __detail::spgemm_fill(a, b, c); })
void multiply_fill(operation_state& state, Alpha alpha, A&& a, B&& b, C&& c) {
  log_trace("");
  __detail::spgemm_fill(a, b, c);

  using V = tensor_scalar_t<C>;
  const V alpha_v = static_cast<V>(alpha);
  if (alpha_v != V(1)) {
    for (auto&& v : __backend::values(c)) {
      v *= alpha_v;
    }
  }
}

} // namespace spblas
