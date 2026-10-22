#pragma once

#include <algorithm>

#include <spblas/backend/backend.hpp>
#include <spblas/concepts.hpp>
#include <spblas/detail/api_concepts.hpp>
#include <spblas/detail/log.hpp>
#include <spblas/detail/operation_state.hpp>
#include <spblas/views/inspectors.hpp>

#include "detail/sparse_output.hpp"
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
//   3-stage variant, separating the pattern from the values so the pattern
//   can be reused while the values of A and B change:
//
//   multiply_compute_symbolic([policy,] state, alpha, A, B, C)
//     Sets state.get_result_nnz(). The user then attaches C's pointer and
//     index arrays (values may follow later).
//   multiply_fill_symbolic([policy,] state, alpha, A, B, C)
//     Fills C's pattern only; C's values are not touched.
//   multiply_numeric([policy,] state, alpha, A, B, C)
//     Fills C's values into its existing pattern. May be called again after
//     the values (not the patterns) of A or B change.
//
//   The reference implementation runs the full kernel in each fill stage
//   (into temporaries from the state's allocator), trading speed for
//   simplicity.
//
//   The supported A / B / C format combinations and their kernels are listed
//   in detail/spgemm/spgemm.hpp. C's pattern is symbolic: entries that are
//   numerically zero (e.g. alpha == 0) are kept.
//
//   Not supported: the [beta, D] term.
//
// The policy-first overloads are in policy_forwarding.hpp.
//

namespace spblas {

namespace __detail {

// C's stored values *= alpha, skipped for alpha == 1.
template <typename Alpha, compressed_output C>
void scale_values_(Alpha alpha, C&& c) {
  using V = tensor_scalar_t<C>;
  const V alpha_v = static_cast<V>(alpha);
  if (alpha_v != V(1)) {
    auto values = c.values();
    const std::size_t n = std::min<std::size_t>(values.size(), c.size());
    for (std::size_t i = 0; i < n; i++) {
      values[i] *= alpha_v;
    }
  }
}

} // namespace __detail

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

  __detail::scale_values_(alpha, c);
}

// 3-stage SpGEMM: compute_symbolic sets state.get_result_nnz().
template <__detail::scalar Alpha, matrix A, matrix B, matrix C>
  requires(__detail::compressed_output<C> &&
           requires(A&& a, B&& b, C&& c) { __detail::spgemm_nnz(a, b, c); })
void multiply_compute_symbolic(operation_state& state, Alpha alpha, A&& a,
                               B&& b, C&& c) {
  log_trace("");
  state.update_impl_(__detail::spgemm_nnz(a, b, c));
}

// 3-stage SpGEMM: fills C's pattern; alpha does not affect it.
template <__detail::scalar Alpha, matrix A, matrix B, matrix C>
  requires(__detail::compressed_output<C> &&
           requires(A&& a, B&& b, C&& c) { __detail::spgemm_fill(a, b, c); })
void multiply_fill_symbolic(operation_state& state, Alpha alpha, A&& a, B&& b,
                            C&& c) {
  log_trace("");
  __detail::fill_pattern_only(
      state, c, "multiply_fill_symbolic",
      [&](auto&& t) { __detail::spgemm_fill(a, b, t); });
}

// 3-stage SpGEMM: fills C's values into the pattern from fill_symbolic.
template <__detail::scalar Alpha, matrix A, matrix B, matrix C>
  requires(__detail::compressed_output<C> &&
           requires(A&& a, B&& b, C&& c) {
             __detail::spgemm_nnz(a, b, c);
             __detail::spgemm_fill(a, b, c);
           })
void multiply_numeric(operation_state& state, Alpha alpha, A&& a, B&& b,
                      C&& c) {
  log_trace("");
  __detail::fill_values_only(state, c, __detail::spgemm_nnz(a, b, c),
                             "multiply_numeric",
                             [&](auto&& t) { __detail::spgemm_fill(a, b, t); });
  __detail::scale_values_(alpha, c);
}

} // namespace spblas
