#pragma once

#include <spblas/backend/backend.hpp>
#include <spblas/concepts.hpp>
#include <spblas/detail/log.hpp>

#include <spblas/algorithms/detail/sparse_dot_product.hpp>
#include <spblas/algorithms/transposed.hpp>
#include <spblas/backend/csr_builder.hpp>
#include <spblas/backend/spa_accumulator.hpp>

namespace spblas {

namespace __detail {

// C = AB
// CSR * CSC -> CSR
// SpGEMM (Inner Product)
template <matrix A, matrix B, matrix C>
  requires(__backend::row_iterable<A> && __backend::column_iterable<B> &&
           __detail::is_csr_view_v<C>)
void spgemm_fill(A&& a, B&& b, C&& c) {
  log_trace("");
  if (__backend::shape(a)[0] != __backend::shape(c)[0] ||
      __backend::shape(b)[1] != __backend::shape(c)[1] ||
      __backend::shape(a)[1] != __backend::shape(b)[0]) {
    throw std::invalid_argument(
        "multiply: matrix dimensions are incompatible.");
  }

  using T = tensor_scalar_t<C>;
  using I = tensor_index_t<C>;

  __backend::spa_accumulator<T, I> dot_product_acc(__backend::shape(a)[1]);
  __backend::spa_accumulator<T, I> c_row(__backend::shape(c)[1]);
  __backend::csr_builder c_builder(c);

  for (auto&& [i, a_row] : __backend::rows(a)) {
    c_row.clear();

    if (!spblas::__ranges::empty(a_row)) {
      for (auto&& [j, b_column] : __backend::columns(b)) {
        if (!spblas::__ranges::empty(b_column)) {
          auto v =
              __detail::sparse_dot_product<T>(dot_product_acc, a_row, b_column);

          if (v.has_value()) {
            c_row[j] += v.value();
          }
        }
      }
      c_row.sort();

      try {
        c_builder.insert_row(i, c_row.get());
      } catch (...) {
        throw std::runtime_error("multiply: SpGEMM ran out of memory.");
      }
    }
  }
  c_builder.finish();
  c.update(c.values(), c.rowptr(), c.colind(), c.shape(),
           c.rowptr()[c.shape()[0]]);
}

// C = AB
// CSR * CSC -> CSR
// SpGEMM (Inner Product)
template <matrix A, matrix B, matrix C>
  requires(__backend::row_iterable<A> && __backend::column_iterable<B> &&
           __detail::is_csr_view_v<C>)
auto spgemm_nnz(A&& a, B&& b, C&& c) {
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

  __backend::spa_set<I> dot_product_acc(__backend::shape(a)[1]);

  for (auto&& [i, a_row] : __backend::rows(a)) {
    if (!spblas::__ranges::empty(a_row)) {
      for (auto&& [j, b_column] : __backend::columns(b)) {
        if (!spblas::__ranges::empty(b_column)) {
          auto v =
              __detail::sparse_intersection(dot_product_acc, a_row, b_column);

          if (v) {
            nnz++;
          }
        }
      }
    }
  }

  return nnz;
}

// C = AB
// CSR * CSC -> CSC
// SpGEMM (Inner Product, transposed)
template <matrix A, matrix B, matrix C>
  requires(__backend::row_iterable<A> && __backend::column_iterable<B> &&
           __detail::is_csc_view_v<C>)
void spgemm_fill(A&& a, B&& b, C&& c) {
  log_trace("");
  if (__backend::shape(a)[0] != __backend::shape(c)[0] ||
      __backend::shape(b)[1] != __backend::shape(c)[1] ||
      __backend::shape(a)[1] != __backend::shape(b)[0]) {
    throw std::invalid_argument(
        "multiply: matrix dimensions are incompatible.");
  }

  return spgemm_fill(transposed(b), transposed(a), transposed(c));
}

// C = AB
// CSR * CSC -> CSC
// SpGEMM (Inner Product, transposed)
template <matrix A, matrix B, matrix C>
  requires(__backend::row_iterable<A> && __backend::column_iterable<B> &&
           __detail::is_csc_view_v<C>)
auto spgemm_nnz(A&& a, B&& b, C&& c) {
  log_trace("");
  if (__backend::shape(a)[0] != __backend::shape(c)[0] ||
      __backend::shape(b)[1] != __backend::shape(c)[1] ||
      __backend::shape(a)[1] != __backend::shape(b)[0]) {
    throw std::invalid_argument(
        "multiply: matrix dimensions are incompatible.");
  }

  return spgemm_nnz(transposed(b), transposed(a), transposed(c));
}

} // namespace __detail

} // namespace spblas
