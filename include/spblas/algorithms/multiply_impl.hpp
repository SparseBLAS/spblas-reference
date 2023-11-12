#pragma once

#include <spblas/backend/backend.hpp>
#include <spblas/concepts.hpp>

#include <spblas/backend/csr_builder.hpp>
#include <spblas/backend/spa_accumulator.hpp>
#include <spblas/detail/operation_info_t.hpp>

#include <algorithm>

namespace spblas {

// SpMV
template <matrix A, vector B, vector C>
void multiply(A&& a, B&& b, C&& c) {
  if (__backend::shape(a)[0] != __backend::shape(c) ||
      __backend::shape(a)[1] != __backend::shape(b)) {
    throw std::invalid_argument(
        "multiply: matrix and vector dimensions are incompatible.");
  }

  __backend::for_each(a, [&](auto&& e) {
    auto&& [idx, a_v] = e;
    auto&& [i, k] = idx;
    c[i] += a_v * b[k];
  });
}

// SpMM
template <matrix A, matrix B, matrix C>
  requires(__backend::lookupable<B> && __backend::lookupable<C>)
void multiply(A&& a, B&& b, C&& c) {
  if (__backend::shape(a)[0] != __backend::shape(c)[0] ||
      __backend::shape(b)[1] != __backend::shape(c)[1] ||
      __backend::shape(a)[1] != __backend::shape(b)[0]) {
    throw std::invalid_argument(
        "multiply: matrix dimensions are incompatible.");
  }

  __backend::for_each(a, [&](auto&& e) {
    auto&& [idx, a_v] = e;
    auto&& [i, k] = idx;
    for (std::size_t j = 0; j < __backend::shape(b)[1]; j++) {
      __backend::lookup(c, i, j) += a_v * __backend::lookup(b, k, j);
    }
  });
}

// SpGEMM (Gustavson's Algorithm)
template <matrix A, matrix B, matrix C>
  requires(__backend::row_iterable<A> && __backend::row_iterable<B> &&
           __detail::is_csr_view_v<C>)
void multiply(A&& a, B&& b, C&& c) {
  if (__backend::shape(a)[0] != __backend::shape(c)[0] ||
      __backend::shape(b)[1] != __backend::shape(c)[1] ||
      __backend::shape(a)[1] != __backend::shape(b)[0]) {
    throw std::invalid_argument(
        "multiply: matrix dimensions are incompatible.");
  }

  using T = tensor_scalar_t<C>;

  __backend::spa_accumulator<T> c_row(__backend::shape(c)[1]);
  __backend::csr_builder c_builder(c);

  for (auto&& [i, a_row] : __backend::rows(a)) {
    c_row.clear();
    auto&& b_rows = __backend::rows(b);
    for (auto&& [k, a_v] : a_row) {
      for (auto&& [j, b_v] : std::get<1>(__backend::rows(b)[k])) {
        c_row[j] += a_v * b_v;
      }
    }
    c_row.sort();

    try {
      c_builder.insert_row(i, c_row.get());
    } catch (...) {
      throw std::runtime_error("matrix: ran out of memory.  CSR output view "
                               "has insufficient memory.");
    }
  }
}

// SpGEMM (Gustavson's Algorithm)
template <matrix A, matrix B, matrix C>
  requires(__backend::row_iterable<A> && __backend::row_iterable<B> &&
           __detail::is_csr_view_v<C>)
operation_info_t multiply_inspect(A&& a, B&& b, C&& c) {
  if (__backend::shape(a)[0] != __backend::shape(c)[0] ||
      __backend::shape(b)[1] != __backend::shape(c)[1] ||
      __backend::shape(a)[1] != __backend::shape(b)[0]) {
    throw std::invalid_argument(
        "multiply: matrix dimensions are incompatible.");
  }

  using T = tensor_scalar_t<C>;
  using I = tensor_index_t<C>;

  std::size_t nnz = 0;
  __backend::spa_set<std::size_t> c_row(__backend::shape(c)[1]);

  for (auto&& [i, a_row] : __backend::rows(a)) {
    c_row.clear();
    auto&& b_rows = __backend::rows(b);
    for (auto&& [k, a_v] : a_row) {
      for (auto&& [j, b_v] : std::get<1>(__backend::rows(b)[k])) {
        c_row.insert(j);
      }
    }
    nnz += c_row.size();
  }

  return operation_info_t{__backend::shape(c), nnz};
}

template <matrix A, matrix B, matrix C>
void multiply_execute(operation_info_t info, A&& a, B&& b, C&& c) {
  multiply(a, b, c);
}

} // namespace spblas
