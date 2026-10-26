#pragma once

#include <memory_resource>

#include <spblas/backend/backend.hpp>
#include <spblas/concepts.hpp>
#include <spblas/detail/log.hpp>

#include <spblas/algorithms/detail/sparse_dot_product.hpp>
#include <spblas/backend/csr_builder.hpp>
#include <spblas/backend/hash_accumulator.hpp>
#include <spblas/views/view.hpp>

namespace spblas {

namespace __detail {

// C = AB
// CSC * CSR -> CSR
// SpGEMM (Outer Product)
template <matrix A, matrix B, matrix C>
  requires(__backend::column_iterable<A> && __backend::row_iterable<B> &&
           __detail::is_csr_view_v<C>)
void spgemm_fill(A&& a, B&& b, C&& c,
                 std::pmr::polymorphic_allocator<> allocator) {
  log_trace("");
  if (__backend::shape(a)[0] != __backend::shape(c)[0] ||
      __backend::shape(b)[1] != __backend::shape(c)[1] ||
      __backend::shape(a)[1] != __backend::shape(b)[0]) {
    throw std::invalid_argument(
        "multiply: matrix dimensions are incompatible.");
  }

  using T = tensor_scalar_t<C>;
  using I = tensor_index_t<C>;

  std::pmr::vector<__backend::hash_accumulator<T, I>> row_accumulators(
      allocator);

  for (std::size_t i = 0; i < __backend::shape(c)[0]; i++) {
    row_accumulators.emplace_back(__backend::shape(c)[1], allocator);
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
auto spgemm_nnz(A&& a, B&& b, C&& c,
                std::pmr::polymorphic_allocator<> allocator) {
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

  std::pmr::vector<__backend::hash_accumulator<T, I>> row_accumulators(
      allocator);

  for (std::size_t i = 0; i < __backend::shape(c)[0]; i++) {
    row_accumulators.emplace_back(__backend::shape(c)[1], allocator);
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

  return nnz;
}

// C = AB
// CSC * CSR -> CSC
// SpGEMM (Outer Product, transposed)
template <matrix A, matrix B, matrix C>
  requires(__backend::column_iterable<A> && __backend::row_iterable<B> &&
           __detail::is_csc_view_v<C>)
void spgemm_fill(A&& a, B&& b, C&& c,
                 std::pmr::polymorphic_allocator<> allocator) {
  log_trace("");
  if (__backend::shape(a)[0] != __backend::shape(c)[0] ||
      __backend::shape(b)[1] != __backend::shape(c)[1] ||
      __backend::shape(a)[1] != __backend::shape(b)[0]) {
    throw std::invalid_argument(
        "multiply: matrix dimensions are incompatible.");
  }
  // C^T = B^T A^T, with C (a wrapper) read as the CSR arrays of C^T.
  spgemm_fill(spblas::view::transpose(b), spblas::view::transpose(a),
              spblas::view::transpose(c), allocator);
}

// C = AB
// CSC * CSR -> CSC
// SpGEMM (Outer Product, transposed)
template <matrix A, matrix B, matrix C>
  requires(__backend::column_iterable<A> && __backend::row_iterable<B> &&
           __detail::is_csc_view_v<C>)
auto spgemm_nnz(A&& a, B&& b, C&& c,
                std::pmr::polymorphic_allocator<> allocator) {
  log_trace("");
  if (__backend::shape(a)[0] != __backend::shape(c)[0] ||
      __backend::shape(b)[1] != __backend::shape(c)[1] ||
      __backend::shape(a)[1] != __backend::shape(b)[0]) {
    throw std::invalid_argument(
        "multiply: matrix dimensions are incompatible.");
  }

  // C^T = B^T A^T, with C (a wrapper) read as the CSR arrays of C^T.
  return spgemm_nnz(spblas::view::transpose(b), spblas::view::transpose(a),
                    spblas::view::transpose(c), allocator);
}

} // namespace __detail

} // namespace spblas
