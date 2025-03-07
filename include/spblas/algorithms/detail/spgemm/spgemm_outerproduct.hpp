#pragma once

#include <spblas/backend/backend.hpp>
#include <spblas/concepts.hpp>
#include <spblas/detail/log.hpp>

#include <spblas/algorithms/detail/sparse_dot_product.hpp>
#include <spblas/algorithms/transposed.hpp>
#include <spblas/backend/csr_builder.hpp>
#include <spblas/backend/hash_accumulator.hpp>
#include <spblas/detail/operation_info_t.hpp>

namespace spblas {

// C = AB
// CSC * CSR -> CSR
// SpGEMM (Outer Product)
template <matrix A, matrix B, matrix C>
  requires(__backend::column_iterable<A> && __backend::row_iterable<B> &&
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

  std::vector<__backend::hash_accumulator<T, I>> row_accumulators;

  for (std::size_t i = 0; i < __backend::shape(c)[0]; i++) {
    row_accumulators.emplace_back(__backend::shape(c)[1]);
  }

  for (std::size_t k = 0; k < __backend::shape(a)[1]; k++) {
    auto&& a_vec = __backend::lookup_column(a, k);
    auto&& b_vec = __backend::lookup_row(b, k);

    for (auto&& [i, a_v] : a_vec) {
      for (auto&& [j, b_v] : b_vec) {
        row_accumulators[i][j] += a_v * b_v;
      }
    }
  }

  __backend::csr_builder c_builder(c);

  for (std::size_t i = 0; i < row_accumulators.size(); i++) {
    auto&& c_row = row_accumulators[i];

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

template <matrix A, matrix B, matrix C>
  requires(__backend::column_iterable<A> && __backend::row_iterable<B> &&
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

  std::vector<__backend::hash_accumulator<T, I>> row_accumulators;

  for (std::size_t i = 0; i < __backend::shape(c)[0]; i++) {
    row_accumulators.emplace_back(__backend::shape(c)[1]);
  }

  O nnz = 0;

  for (std::size_t k = 0; k < __backend::shape(a)[1]; k++) {
    auto&& a_vec = __backend::lookup_column(a, k);
    auto&& b_vec = __backend::lookup_row(b, k);

    for (auto&& [i, a_v] : a_vec) {
      for (auto&& [j, b_v] : b_vec) {
        row_accumulators[i][j] += a_v * b_v;
      }
    }
  }

  for (auto&& row_acc : row_accumulators) {
    nnz += row_acc.size();
  }

  return operation_info_t{__backend::shape(c), nnz};
}

// C = AB
// CSC * CSR -> CSC
// SpGEMM (Outer Product, transposed)
template <matrix A, matrix B, matrix C>
  requires(__backend::column_iterable<A> && __backend::row_iterable<B> &&
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
// CSC * CSR -> CSC
// SpGEMM (Outer Product, transposed)
template <matrix A, matrix B, matrix C>
  requires(__backend::column_iterable<A> && __backend::row_iterable<B> &&
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

} // namespace spblas
