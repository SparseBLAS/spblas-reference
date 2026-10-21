#pragma once

#include <algorithm>
#include <cmath>
#include <memory_resource>
#include <utility>
#include <vector>

#include <spblas/backend/backend.hpp>
#include <spblas/concepts.hpp>
#include <spblas/detail/api_concepts.hpp>
#include <spblas/detail/log.hpp>
#include <spblas/detail/operation_state.hpp>

//
// Reference matrix norms (SparseBLAS paper)
//
//   matrix_inf_norm([policy,] state, A)   max_i sum_j |a_ij|
//   matrix_frob_norm([policy,] state, A)  sqrt(sum_ij |a_ij|^2)
//
//   A: any matrix the CPOs can iterate -- CSR, CSC, a matrix_opt of either,
//   and view::transpose / view::conjugate of those (so the 1-norm of A is
//   matrix_inf_norm(state, view::transpose(A))). Each stored entry counts
//   once; duplicate entries are not combined first. Returns the real type
//   of A's values.
//
//   The Frobenius norm uses LAPACK's scaled sum of squares (lassq), so it
//   neither overflows nor underflows for representable results.
//

namespace spblas {

namespace __detail {

template <typename M>
using norm_t = decltype(std::abs(std::declval<tensor_scalar_t<M>>()));

} // namespace __detail

template <matrix A>
  requires(__backend::row_iterable<A> || __backend::column_iterable<A>)
__detail::norm_t<A> matrix_inf_norm(operation_state& state, A&& a) {
  log_trace("");
  using R = __detail::norm_t<A>;

  std::pmr::vector<R> row_sums(__backend::shape(a)[0], R(0),
                               state.get_allocator());
  __backend::for_each(a, [&](auto&& e) {
    auto&& [idx, v] = e;
    auto&& [i, j] = idx;
    row_sums[i] += std::abs(v);
  });

  R norm = 0;
  for (auto&& s : row_sums) {
    norm = std::max(norm, s);
  }
  return norm;
}

template <matrix A>
  requires(__backend::row_iterable<A> || __backend::column_iterable<A>)
__detail::norm_t<A> matrix_frob_norm(operation_state& state, A&& a) {
  log_trace("");
  using R = __detail::norm_t<A>;

  // norm = scale * sqrt(sum_of_squares), with every |a_ij| / scale <= 1.
  R scale = 0;
  R sum_of_squares = 1;
  auto accumulate = [&](R abs_v) {
    if (abs_v == R(0)) {
      return;
    }
    if (scale < abs_v) {
      R ratio = scale / abs_v;
      sum_of_squares = 1 + sum_of_squares * ratio * ratio;
      scale = abs_v;
    } else {
      R ratio = abs_v / scale;
      sum_of_squares += ratio * ratio;
    }
  };

  __backend::for_each(a, [&](auto&& e) {
    auto&& [idx, v] = e;
    if constexpr (__detail::is_std_complex_v<decltype(v)>) {
      accumulate(std::abs(v.real()));
      accumulate(std::abs(v.imag()));
    } else {
      accumulate(std::abs(v));
    }
  });

  return scale * std::sqrt(sum_of_squares);
}

// Optional leading execution policy. The reference implementation runs
// sequentially and ignores it.
template <__detail::execution_policy P, typename... Args>
auto matrix_inf_norm(P&& policy, operation_state& state, Args&&... args) {
  return matrix_inf_norm(state, std::forward<Args>(args)...);
}

template <__detail::execution_policy P, typename... Args>
auto matrix_frob_norm(P&& policy, operation_state& state, Args&&... args) {
  return matrix_frob_norm(state, std::forward<Args>(args)...);
}

} // namespace spblas
