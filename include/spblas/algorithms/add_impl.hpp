#pragma once

#include <spblas/backend/backend.hpp>
#include <spblas/backend/csr_builder.hpp>
#include <spblas/backend/spa_accumulator.hpp>
#include <spblas/concepts.hpp>
#include <spblas/detail/api_concepts.hpp>
#include <spblas/detail/operation_info_t.hpp>
#include <spblas/detail/operation_state.hpp>

namespace spblas {

template <vector A, vector B, vector C>
void add(A&& a, B&& b, C&& c) {
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
  if (__backend::shape(a) != __backend::shape(b) ||
      __backend::shape(b) != __backend::shape(c)) {
    throw std::invalid_argument("add: matrix dimensions are incompatible.");
  }

  using T = tensor_scalar_t<C>;
  using I = tensor_index_t<C>;

  __backend::spa_accumulator<T, I> c_row(__backend::shape(c)[1]);
  __backend::csr_builder c_builder(c);

  for (I i = 0; i < __backend::shape(c)[0]; i++) {
    c_row.clear();

    for (auto&& [j, v] : __backend::lookup_row(a, i)) {
      c_row[j] += v;
    }

    for (auto&& [j, v] : __backend::lookup_row(b, i)) {
      c_row[j] += v;
    }

    c_row.sort();

    try {
      c_builder.insert_row(i, c_row.get());
    } catch (...) {
      throw std::runtime_error("add: ran out of memory.  CSR output view "
                               "has insufficient memory.");
    }
  }
  c.update(c.values(), c.rowptr(), c.colind(), c.shape(),
           c.rowptr()[c.shape()[0]]);
}

template <matrix A, matrix B, matrix C>
  requires(__backend::row_lookupable<A> && __backend::row_lookupable<B> &&
           __backend::row_lookupable<C>)
operation_info_t add_inspect(A&& a, B&& b, C&& c) {
  if (__backend::shape(a) != __backend::shape(b) ||
      __backend::shape(b) != __backend::shape(c)) {
    throw std::invalid_argument("add: matrix dimensions are incompatible.");
  }

  using I = tensor_index_t<C>;

  std::size_t nnz = 0;
  __backend::spa_set<I> c_row(__backend::shape(c)[1]);

  for (I i = 0; i < __backend::shape(c)[0]; i++) {
    c_row.clear();

    for (auto&& [j, _] : __backend::lookup_row(a, i)) {
      c_row.insert(j);
    }

    for (auto&& [j, _] : __backend::lookup_row(b, i)) {
      c_row.insert(j);
    }

    nnz += c_row.size();
  }

  return operation_info_t{__backend::shape(c), index_t(nnz)};
}

template <matrix A, matrix B, matrix C>
void add_compute(operation_info_t& info, A&& a, B&& b, C&& c) {
  add(a, b, c);
}

//
// operation_state API (SparseBLAS paper)
//
//   add_inspect([policy,] state, A, B, C)
//   add_compute([policy,] state, A, B, C)   sets state.get_result_nnz()
//   add_fill([policy,] state, A, B, C)      C = A + B
//
// Note the stage names differ from the legacy API above, where add_inspect
// counts and add_compute fills.
//

// The reference implementation has nothing to prepare.
template <matrix A, matrix B, matrix C>
void add_inspect(operation_state& state, A&& a, B&& b, C&& c) {}

template <matrix A, matrix B, matrix C>
  requires(__backend::row_lookupable<A> && __backend::row_lookupable<B> &&
           __detail::is_csr_wrapper_v<C>)
void add_compute(operation_state& state, A&& a, B&& b, C&& c) {
  auto info = add_inspect(a, b, c);
  state.update_impl_(info.result_nnz());
}

template <matrix A, matrix B, matrix C>
  requires(__backend::row_iterable<A> && __backend::row_iterable<B> &&
           __detail::is_csr_wrapper_v<C>)
void add_fill(operation_state& state, A&& a, B&& b, C&& c) {
  add(a, b, c);
}

// Optional leading execution policy. The reference implementation runs
// sequentially and ignores it.
template <__detail::execution_policy P, typename... Args>
void add_inspect(P&& policy, operation_state& state, Args&&... args) {
  add_inspect(state, std::forward<Args>(args)...);
}

template <__detail::execution_policy P, typename... Args>
void add_compute(P&& policy, operation_state& state, Args&&... args) {
  add_compute(state, std::forward<Args>(args)...);
}

template <__detail::execution_policy P, typename... Args>
void add_fill(P&& policy, operation_state& state, Args&&... args) {
  add_fill(state, std::forward<Args>(args)...);
}

} // namespace spblas
