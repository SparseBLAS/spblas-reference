#pragma once

#include <spblas/backend/backend.hpp>
#include <spblas/concepts.hpp>
#include <spblas/detail/log.hpp>

#include <spblas/algorithms/transposed.hpp>
#include <spblas/backend/csr_builder.hpp>
#include <spblas/backend/spa_accumulator.hpp>
#include <spblas/detail/operation_info_t.hpp>

namespace spblas {

// C = AB
// CSR * CSR -> CSR
// SpGEMM (Gustavson's Algorithm)
template <matrix A, matrix B, matrix C>
  requires(__backend::row_iterable<A> && __backend::row_iterable<B> &&
           __detail::is_csr_view_v<C>)
void multiply(A&& a, B&& b, C&& c) {
  log_trace("");
  if (__backend::shape(a)[0] != __backend::shape(c)[0] ||
      __backend::shape(b)[1] != __backend::shape(c)[1] ||
      __backend::shape(a)[1] != __backend::shape(b)[0]) {
    throw std::invalid_argument(
        "multiply: matrix dimensions are incompatible.");
  }

  using T = tensor_scalar_t<C>;
  using I = tensor_index_t<C>;

  __backend::spa_accumulator<T, I> c_row(__backend::shape(c)[1]);
  __backend::csr_builder c_builder(c);

  for (auto&& [i, a_row] : __backend::rows(a)) {
    c_row.clear();
    for (auto&& [k, a_v] : a_row) {
      for (auto&& [j, b_v] : __backend::lookup_row(b, k)) {
        c_row[j] += a_v * b_v;
      }
    }
    c_row.sort();

    try {
      c_builder.insert_row(i, c_row.get());
    } catch (...) {
      throw std::runtime_error("multiply: SpGEMM ran out of memory.");
    }
  }
  c.update(c.values(), c.rowptr(), c.colind(), c.shape(),
           c.rowptr()[c.shape()[0]]);
}

// C = AB
// CSR * CSR -> CSR
// SpGEMM (Gustavson's Algorithm)
template <matrix A, matrix B, matrix C>
  requires(__backend::row_iterable<A> && __backend::row_iterable<B> &&
           __detail::is_csr_view_v<C>)
operation_info_t multiply_compute(A&& a, B&& b, C&& c) {
  log_trace("");
  if (__backend::shape(a)[0] != __backend::shape(c)[0] ||
      __backend::shape(b)[1] != __backend::shape(c)[1] ||
      __backend::shape(a)[1] != __backend::shape(b)[0]) {
    throw std::invalid_argument(
        "multiply: matrix dimensions are incompatible.");
  }

  using T = tensor_scalar_t<C>;
  using I = tensor_index_t<C>;
  using O = tensor_offset_t<C>;

  O nnz = 0;
  __backend::spa_set<I> c_row(__backend::shape(c)[1]);

  for (auto&& [i, a_row] : __backend::rows(a)) {
    c_row.clear();

    for (auto&& [k, a_v] : a_row) {
      for (auto&& [j, b_v] : __backend::lookup_row(b, k)) {
        c_row.insert(j);
      }
    }

    nnz += c_row.size();
  }

  return operation_info_t{__backend::shape(c), nnz};
}

// C = AB
// CSC * CSC -> CSC
// SpGEMM (Gustavson's Algorithm, transposed)
template <matrix A, matrix B, matrix C>
  requires(__backend::column_iterable<A> && __backend::column_iterable<B> &&
           __detail::is_csc_view_v<C>)
void multiply(A&& a, B&& b, C&& c) {
  log_trace("");
  if (__backend::shape(a)[0] != __backend::shape(c)[0] ||
      __backend::shape(b)[1] != __backend::shape(c)[1] ||
      __backend::shape(a)[1] != __backend::shape(b)[0]) {
    throw std::invalid_argument(
        "multiply: matrix dimensions are incompatible.");
  }
  multiply(transposed(b), transposed(a), transposed(c));
}

// C = AB
// CSC * CSC -> CSC
// SpGEMM (Gustavson's Algorithm, transposed)
template <matrix A, matrix B, matrix C>
  requires(__backend::column_iterable<A> && __backend::column_iterable<B> &&
           __detail::is_csc_view_v<C>)
operation_info_t multiply_compute(A&& a, B&& b, C&& c) {
  log_trace("");
  if (__backend::shape(a)[0] != __backend::shape(c)[0] ||
      __backend::shape(b)[1] != __backend::shape(c)[1] ||
      __backend::shape(a)[1] != __backend::shape(b)[0]) {
    throw std::invalid_argument(
        "multiply: matrix dimensions are incompatible.");
  }

  auto info = multiply_compute(transposed(b), transposed(a), transposed(c));
  info.update_impl_({info.result_shape()[1], info.result_shape()[0]},
                    info.result_nnz());
  return info;
}

// C = AB
// CSR * CSR -> CSC
// SpGEMM (Gustavson's Algorithm, scattered)
template <matrix A, matrix B, matrix C>
  requires(__backend::row_iterable<A> && __backend::row_iterable<B> &&
           __detail::is_csc_view_v<C>)
void multiply(A&& a, B&& b, C&& c) {
  log_trace("");
  if (__backend::shape(a)[0] != __backend::shape(c)[0] ||
      __backend::shape(b)[1] != __backend::shape(c)[1] ||
      __backend::shape(a)[1] != __backend::shape(b)[0]) {
    throw std::invalid_argument(
        "multiply: matrix dimensions are incompatible.");
  }

  using T = tensor_scalar_t<C>;
  using I = tensor_index_t<C>;

  __backend::spa_accumulator<T, I> c_row(__backend::shape(c)[1]);

  std::vector<std::vector<std::pair<I, T>>> columns(__backend::shape(c)[1]);

  for (auto&& [i, a_row] : __backend::rows(a)) {
    c_row.clear();
    for (auto&& [k, a_v] : a_row) {
      for (auto&& [j, b_v] : __backend::lookup_row(b, k)) {
        c_row[j] += a_v * b_v;
      }
    }
    for (auto&& [j, v] : c_row.get()) {
      columns[j].push_back({i, v});
    }
  }

  __backend::csc_builder c_builder(c);

  for (std::size_t j = 0; j < columns.size(); j++) {
    auto&& column = columns[j];
    std::sort(column.begin(), column.end(),
              [](auto&& a, auto&& b) { return a.first < b.first; });

    try {
      c_builder.insert_column(j, column);
    } catch (...) {
      throw std::runtime_error("multiply: SpGEMM ran out of memory.");
    }
  }
  c.update(c.values(), c.colptr(), c.rowind(), c.shape(),
           c.colptr()[c.shape()[1]]);
}

// C = AB
// CSR * CSR -> CSC
// SpGEMM (Gustavson's Algorithm, scattered)
template <matrix A, matrix B, matrix C>
  requires(__backend::row_iterable<A> && __backend::row_iterable<B> &&
           __detail::is_csc_view_v<C>)
operation_info_t multiply_compute(A&& a, B&& b, C&& c) {
  log_trace("");
  if (__backend::shape(a)[0] != __backend::shape(c)[0] ||
      __backend::shape(b)[1] != __backend::shape(c)[1] ||
      __backend::shape(a)[1] != __backend::shape(b)[0]) {
    throw std::invalid_argument(
        "multiply: matrix dimensions are incompatible.");
  }

  using T = tensor_scalar_t<C>;
  using I = tensor_index_t<C>;
  using O = tensor_offset_t<C>;

  O nnz = 0;
  __backend::spa_set<I> c_row(__backend::shape(c)[1]);

  for (auto&& [i, a_row] : __backend::rows(a)) {
    c_row.clear();

    for (auto&& [k, a_v] : a_row) {
      for (auto&& [j, b_v] : __backend::lookup_row(b, k)) {
        c_row.insert(j);
      }
    }

    nnz += c_row.size();
  }

  return operation_info_t{__backend::shape(c), nnz};
}

} // namespace spblas
