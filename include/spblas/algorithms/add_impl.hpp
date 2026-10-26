#pragma once

#include <spblas/algorithms/detail/spadd.hpp>
#include <spblas/algorithms/detail/sparse_output.hpp>
#include <spblas/backend/backend.hpp>
#include <spblas/backend/csr_builder.hpp>
#include <spblas/backend/spa_accumulator.hpp>
#include <spblas/concepts.hpp>
#include <spblas/detail/api_concepts.hpp>
#include <spblas/detail/log.hpp>
#include <spblas/detail/operation_info_t.hpp>
#include <spblas/detail/operation_state.hpp>

namespace spblas {

template <vector A, vector B, vector C>
void add(A&& a, B&& b, C&& c) {
  log_trace("");
  if (__backend::shape(a) != __backend::shape(b) ||
      __backend::shape(b) != __backend::shape(c)) {
    throw std::invalid_argument("add: vector dimensions are incompatible.");
  }

  __backend::for_each(c, [&](auto&& e) {
    auto&& [i, c_v] = e;
    c_v = __backend::lookup(a, i) + __backend::lookup(b, i);
  });
}

template <matrix A, matrix B, matrix C>
  requires(__backend::lookupable<A> && __backend::lookupable<B> &&
           __backend::lookupable<C>)
void add(A&& a, B&& b, C&& c) {
  log_trace("");
  if (__backend::shape(a) != __backend::shape(b) ||
      __backend::shape(b) != __backend::shape(c)) {
    throw std::invalid_argument("add: matrix dimensions are incompatible.");
  }

  for (std::size_t i = 0; i < __backend::shape(c)[0]; i++) {
    for (std::size_t j = 0; j < __backend::shape(c)[1]; j++) {
      __backend::lookup(c, i, j) =
          __backend::lookup(a, i, j) + __backend::lookup(b, i, j);
    }
  }
}

template <matrix A, matrix B, matrix C>
  requires(__backend::row_iterable<A> && __backend::row_iterable<B> &&
           __detail::is_csr_view_v<C>)
void add(A&& a, B&& b, C&& c) {
  log_trace("");
  __detail::spadd_fill(a, b, c, __detail::default_memory_resource());
}

template <matrix A, matrix B, matrix C>
  requires(__backend::row_lookupable<A> && __backend::row_lookupable<B> &&
           __backend::row_lookupable<C>)
operation_info_t add_inspect(A&& a, B&& b, C&& c) {
  log_trace("");
  return operation_info_t{__backend::shape(c),
                          index_t(__detail::spadd_nnz(
                              a, b, c, __detail::default_memory_resource()))};
}

template <matrix A, matrix B, matrix C>
void add_compute(operation_info_t& info, A&& a, B&& b, C&& c) {
  log_trace("");
  add(a, b, c);
}

//
// operation_state API (SparseBLAS paper)
//
//   add_inspect([policy,] state, A, B, C)
//   add_compute([policy,] state, A, B, C)   sets state.get_result_nnz()
//   add_fill([policy,] state, A, B, C)      C = A + B
//
//   3-stage variant, separating the pattern from the values:
//
//   add_compute_symbolic([policy,] state, A, B, C)  sets get_result_nnz()
//   add_fill_symbolic([policy,] state, A, B, C)     C's pattern only
//   add_numeric([policy,] state, A, B, C)           C's values, into the
//                                                   existing pattern
//
//   A, B, C all CSR or all CSC (matrix_opt inputs work too); kernels in
//   detail/spadd.hpp. The reference implementation runs the full kernel in
//   each 3-stage fill stage (into temporaries from the state's allocator).
//   The kernel sorts each row (column) of C, so the fill stages set
//   C.sort_state() to `sorted`.
//
// Note the stage names differ from the legacy API above, where add_inspect
// counts and add_compute fills.
//

// The reference implementation has nothing to prepare.
template <matrix A, matrix B, matrix C>
void add_inspect(operation_state& state, A&& a, B&& b, C&& c) {
  log_trace("");
}

template <matrix A, matrix B, matrix C>
  requires(__detail::spadd_by_rows<A, B, C> ||
           __detail::spadd_by_columns<A, B, C>)
void add_compute(operation_state& state, A&& a, B&& b, C&& c) {
  log_trace("");
  state.update_impl_(__detail::spadd_nnz(a, b, c, state.get_allocator()));
}

template <matrix A, matrix B, matrix C>
  requires(__detail::spadd_by_rows<A, B, C> ||
           __detail::spadd_by_columns<A, B, C>)
void add_fill(operation_state& state, A&& a, B&& b, C&& c) {
  log_trace("");
  __detail::spadd_fill(a, b, c, state.get_allocator());
  __detail::set_output_sort_state(c, sort_state::sorted);
}

template <matrix A, matrix B, matrix C>
  requires(__detail::spadd_by_rows<A, B, C> ||
           __detail::spadd_by_columns<A, B, C>)
void add_compute_symbolic(operation_state& state, A&& a, B&& b, C&& c) {
  log_trace("");
  state.update_impl_(__detail::spadd_nnz(a, b, c, state.get_allocator()));
}

template <matrix A, matrix B, matrix C>
  requires(__detail::spadd_by_rows<A, B, C> ||
           __detail::spadd_by_columns<A, B, C>)
void add_fill_symbolic(operation_state& state, A&& a, B&& b, C&& c) {
  log_trace("");
  __detail::fill_pattern_only(state, c, "add_fill_symbolic", [&](auto&& t) {
    __detail::spadd_fill(a, b, t, state.get_allocator());
  });
  __detail::set_output_sort_state(c, sort_state::sorted);
}

template <matrix A, matrix B, matrix C>
  requires(__detail::spadd_by_rows<A, B, C> ||
           __detail::spadd_by_columns<A, B, C>)
void add_numeric(operation_state& state, A&& a, B&& b, C&& c) {
  log_trace("");
  __detail::fill_values_only(
      state, c, __detail::spadd_nnz(a, b, c, state.get_allocator()),
      "add_numeric",
      [&](auto&& t) { __detail::spadd_fill(a, b, t, state.get_allocator()); });
}

template <__detail::execution_policy P, typename... Args>
void add_inspect(P&& policy, operation_state& state, Args&&... args) {
  log_trace("");
  add_inspect(state, std::forward<Args>(args)...);
}

template <__detail::execution_policy P, typename... Args>
void add_compute(P&& policy, operation_state& state, Args&&... args) {
  log_trace("");
  add_compute(state, std::forward<Args>(args)...);
}

template <__detail::execution_policy P, typename... Args>
void add_fill(P&& policy, operation_state& state, Args&&... args) {
  log_trace("");
  add_fill(state, std::forward<Args>(args)...);
}

template <__detail::execution_policy P, typename... Args>
void add_compute_symbolic(P&& policy, operation_state& state, Args&&... args) {
  log_trace("");
  add_compute_symbolic(state, std::forward<Args>(args)...);
}

template <__detail::execution_policy P, typename... Args>
void add_fill_symbolic(P&& policy, operation_state& state, Args&&... args) {
  log_trace("");
  add_fill_symbolic(state, std::forward<Args>(args)...);
}

template <__detail::execution_policy P, typename... Args>
void add_numeric(P&& policy, operation_state& state, Args&&... args) {
  log_trace("");
  add_numeric(state, std::forward<Args>(args)...);
}

} // namespace spblas
