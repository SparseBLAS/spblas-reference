#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory_resource>
#include <utility>
#include <vector>

#include <spblas/backend/backend.hpp>
#include <spblas/concepts.hpp>
#include <spblas/detail/api_concepts.hpp>
#include <spblas/detail/log.hpp>
#include <spblas/detail/operation_state.hpp>
#include <spblas/views/view.hpp>

//
// Reference matrix norms (SparseBLAS paper)
//
//   matrix_inf_norm([policy,] state, A)   max_i sum_j |a_ij|
//   matrix_one_norm([policy,] state, A)   max_j sum_i |a_ij|, i.e. the
//                                         inf-norm of A^T (not in the paper)
//   matrix_frob_norm([policy,] state, A)  sqrt(sum_ij |a_ij|^2)
//
//   A: any matrix the CPOs can iterate -- CSR, CSC, a matrix_opt of either,
//   and view::transpose / view::conjugate of those. Each stored entry counts
//   once; duplicate entries are not combined first. Returns the real type
//   of A's values.
//
//   The Frobenius norm uses LAPACK's scaled sum of squares (lassq), so it
//   neither overflows nor underflows for representable results.
//
//   A NaN entry makes either norm NaN; otherwise an infinite entry makes it
//   infinite.
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
    if (std::isnan(s)) {
      return s; // std::max would drop it
    }
    norm = std::max(norm, s);
  }
  return norm;
}

// The inf-norm of A^T, so a vendor's inf-norm for transposed views serves
// it too.
template <matrix A>
  requires(__backend::row_iterable<A> || __backend::column_iterable<A>)
__detail::norm_t<A> matrix_one_norm(operation_state& state, A&& a) {
  log_trace("");
  return matrix_inf_norm(state, view::transpose(std::forward<A>(a)));
}

template <matrix A>
  requires(__backend::row_iterable<A> || __backend::column_iterable<A>)
__detail::norm_t<A> matrix_frob_norm(operation_state& state, A&& a) {
  log_trace("");
  using R = __detail::norm_t<A>;

  // norm = scale * sqrt(sum_of_squares), with every finite |a_ij| / scale
  // <= 1. NaN and infinite components are only recorded: scaling by them
  // would turn inf / inf into NaN.
  R scale = 0;
  R sum_of_squares = 1;
  bool has_nan = false;
  bool has_inf = false;
  auto accumulate = [&](R abs_v) {
    if (std::isnan(abs_v)) {
      has_nan = true;
      return;
    }
    if (std::isinf(abs_v)) {
      has_inf = true;
      return;
    }
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

  if (has_nan) {
    return std::numeric_limits<R>::quiet_NaN();
  }
  if (has_inf) {
    return std::numeric_limits<R>::infinity();
  }
  return scale * std::sqrt(sum_of_squares);
}

// Optional leading execution policy. The reference implementation runs
// sequentially and ignores it.
template <__detail::execution_policy P, typename... Args>
auto matrix_inf_norm(P&& policy, operation_state& state, Args&&... args) {
  log_trace("");
  return matrix_inf_norm(state, std::forward<Args>(args)...);
}

template <__detail::execution_policy P, typename... Args>
auto matrix_one_norm(P&& policy, operation_state& state, Args&&... args) {
  log_trace("");
  return matrix_one_norm(state, std::forward<Args>(args)...);
}

template <__detail::execution_policy P, typename... Args>
auto matrix_frob_norm(P&& policy, operation_state& state, Args&&... args) {
  log_trace("");
  return matrix_frob_norm(state, std::forward<Args>(args)...);
}

} // namespace spblas
