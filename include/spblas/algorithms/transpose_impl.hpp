#pragma once

#include <numeric>
#include <stdexcept>

#include <spblas/backend/backend.hpp>
#include <spblas/concepts.hpp>
#include <spblas/detail/api_concepts.hpp>
#include <spblas/detail/operation_info_t.hpp>
#include <spblas/detail/operation_state.hpp>
#include <spblas/detail/view_inspectors.hpp>

namespace spblas {

template <matrix A, matrix B>
operation_info_t transpose_inspect(A&& a, B&& b) {
  return {};
}

namespace __detail {

// B = A^T into B's CSR arrays (sized for A's entries). A is any matrix whose
// entries the CPOs can iterate (CSR, CSC, matrix_opt of either). For
// view::conjugate(A) the entries arrive conjugated, which gives B = A^H.
template <matrix A, matrix B>
  requires((__backend::row_iterable<A> || __backend::column_iterable<A>) &&
           is_csr_wrapper_v<B>)
void transpose_fill(A&& a, B&& b) {
  if (__backend::shape(a)[0] != __backend::shape(b)[1] ||
      __backend::shape(a)[1] != __backend::shape(b)[0]) {
    throw std::invalid_argument(
        "transpose: matrix dimensions are incompatible.");
  }
  if (b.rowptr().size() != __backend::shape(b)[0] + 1) {
    throw std::invalid_argument(
        "transpose: the output's rowptr must have (rows + 1) entries.");
  }
  if (b.values().size() < __backend::size(a) ||
      b.colind().size() < __backend::size(a)) {
    throw std::runtime_error("transpose: Transpose ran out of memory.");
  }
  using O = tensor_offset_t<B>;

  const auto b_rowptr = b.rowptr();
  const auto b_colind = b.colind();
  const auto b_values = b.values();

  // Count entries per output row (input column) at rowptr[j + 1]; after the
  // exclusive scan rowptr[j + 1] is the start of output row j, and placing
  // the entries advances it to the row's end.
  spblas::__ranges::fill(b_rowptr, 0);
  __backend::for_each(a, [&](auto&& e) {
    auto&& [idx, _] = e;
    auto&& [i, j] = idx;
    b_rowptr[j + 1]++;
  });

  std::exclusive_scan(b_rowptr.begin(), b_rowptr.end(), b_rowptr.begin(), O{});

  __backend::for_each(a, [&](auto&& e) {
    auto&& [idx, v] = e;
    auto&& [i, j] = idx;
    const auto out_idx = b_rowptr[j + 1]++;
    b_colind[out_idx] = i;
    b_values[out_idx] = v;
  });

  b.update(b.values(), b.rowptr(), b.colind(), b.shape(),
           b.rowptr()[__backend::shape(b)[0]]);
}

} // namespace __detail

template <matrix A, matrix B>
  requires(__detail::is_csr_view_v<A> && __detail::is_csr_view_v<B>)
void transpose(A&& a, B&& b) {
  __detail::transpose_fill(a, b);
}

template <matrix A, matrix B>
  requires(__detail::is_csr_view_v<A> && __detail::is_csr_view_v<B>)
void transpose(operation_info_t& info, A&& a, B&& b) {
  transpose(std::forward<A>(a), std::forward<B>(b));
}

//
// operation_state API (SparseBLAS paper)
//
//   transpose_inspect([policy,] state, A, B)
//   transpose([policy,] state, A, B)          B = A^T
//   transpose([policy,] state, view::conjugate(A), B)   B = A^H
//
// Single stage: B has the same number of stored entries as A, so the user
// allocates B's arrays up front. A: CSR, CSC, a matrix_opt of either, or a
// view::conjugate of those; B: CSR.
//

namespace __detail {

template <typename A>
concept transpose_input =
    __backend::row_iterable<A> || __backend::column_iterable<A>;

} // namespace __detail

// The reference implementation has nothing to prepare.
template <matrix A, matrix B>
  requires(__detail::transpose_input<A> && __detail::is_csr_wrapper_v<B>)
void transpose_inspect(operation_state& state, A&& a, B&& b) {}

template <matrix A, matrix B>
  requires(__detail::transpose_input<A> && __detail::is_csr_wrapper_v<B>)
void transpose(operation_state& state, A&& a, B&& b) {
  __detail::transpose_fill(a, b);
}

// Optional leading execution policy. The reference implementation runs
// sequentially and ignores it.
template <__detail::execution_policy P, typename... Args>
void transpose_inspect(P&& policy, operation_state& state, Args&&... args) {
  transpose_inspect(state, std::forward<Args>(args)...);
}

template <__detail::execution_policy P, typename... Args>
void transpose(P&& policy, operation_state& state, Args&&... args) {
  transpose(state, std::forward<Args>(args)...);
}

} // namespace spblas
