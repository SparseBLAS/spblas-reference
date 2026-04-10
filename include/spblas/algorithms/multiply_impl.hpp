#pragma once

#include <spblas/backend/backend.hpp>
#include <spblas/concepts.hpp>
#include <spblas/detail/log.hpp>

#include <spblas/algorithms/transposed.hpp>
#include <spblas/backend/csr_builder.hpp>
#include <spblas/backend/spa_accumulator.hpp>
#include <spblas/detail/operation_info_t.hpp>

#include "detail/spgemm/spgemm.hpp"

#include <algorithm>

namespace spblas {

// SpMV inspect
template <matrix A, vector B, vector C>
operation_info_t multiply_inspect(A&& a, B&& b, C&& c) {
  log_trace("");
  return operation_info_t{};
}

// SpMV inspect
template <matrix A, vector B, vector C>
operation_info_t multiply_inspect(operation_info_t& info, A&& a, B&& b, C&& c) {
  log_trace("");
}

// C = AB
// SpMV
template <matrix A, vector B, vector C>
  requires(__backend::lookupable<B> && __backend::lookupable<C>)
void multiply(A&& a, B&& b, C&& c) {
  log_trace("");
  if (__backend::shape(a)[0] != __backend::shape(c) ||
      __backend::shape(a)[1] != __backend::shape(b)) {
    throw std::invalid_argument(
        "multiply: matrix and vector dimensions are incompatible.");
  }

  __backend::for_each(c, [](auto&& e) {
    auto&& [_, v] = e;
    v = 0;
  });

  __backend::for_each(a, [&](auto&& e) {
    auto&& [idx, a_v] = e;
    auto&& [i, k] = idx;
    __backend::lookup(c, i) += a_v * __backend::lookup(b, k);
  });
}

// C = AB
// SpMV with info input
template <matrix A, vector B, vector C>
  requires(__backend::lookupable<B> && __backend::lookupable<C>)
void multiply(operation_info_t& info, A&& a, B&& b, C&& c) {
  log_trace("");
  multiply(std::forward<A>(a), std::forward<B>(b), std::forward<C>(c));
}

// C = AB
// SpMM
template <matrix A, matrix B, matrix C>
  requires(__backend::lookupable<B> && __backend::lookupable<C>)
void multiply(A&& a, B&& b, C&& c) {
  log_trace("");
  if (__backend::shape(a)[0] != __backend::shape(c)[0] ||
      __backend::shape(b)[1] != __backend::shape(c)[1] ||
      __backend::shape(a)[1] != __backend::shape(b)[0]) {
    throw std::invalid_argument(
        "multiply: matrix dimensions are incompatible.");
  }

  // initializes c to zero so we can use += everywhere
  __backend::for_each(c, [](auto&& e) {
    auto&& [_, v] = e;
    v = 0;
  });

  // traverses elements of a and performs appropriate
  // multiplication with B rows
  __backend::for_each(a, [&](auto&& e) {
    auto&& [idx, a_v] = e;
    auto&& [i, k] = idx;
    for (std::size_t j = 0; j < __backend::shape(b)[1]; j++) {
      __backend::lookup(c, i, j) += a_v * __backend::lookup(b, k, j);
    }
  });
}

// C = AB
// SpMM with info
template <matrix A, matrix B, matrix C>
  requires(__backend::lookupable<B> && __backend::lookupable<C>)
void multiply(operation_info_t& info, A&& a, B&& b, C&& c) {
  log_trace("");
  multiply(std::forward<A>(a), std::forward<B>(b), std::forward<C>(c));
}

// C = AB
// SpMM or SpGEMM multiply_inspect variants end up here
template <matrix A, matrix B, matrix C>
operation_info_t multiply_inspect(A&& a, B&& b, C&& c) {
  log_trace("");
  return operation_info_t{};
}

// C = AB
// SpMM or SpGEMM multiply_inspect variants end up here
template <matrix A, matrix B, matrix C>
void multiply_inspect(operation_info_t& info, A&& a, B&& b, C&& c) {
  log_trace("");
};

// C = AB
// SpGEMM compute stage with CSR output
template <matrix A, matrix B, matrix C>
  requires(__backend::row_iterable<A> && __backend::row_iterable<B> &&
           __detail::is_csr_view_v<C>)
void multiply_compute(operation_info_t& info, A&& a, B&& b, C&& c) {
  log_trace("");
  auto new_info = multiply_compute(std::forward<A>(a), std::forward<B>(b),
                                   std::forward<C>(c));
  info.update_impl_(new_info.result_shape(), new_info.result_nnz());
}

// C = AB
// SpGEMM compute stage with CSC output
template <matrix A, matrix B, matrix C>
  requires(__backend::column_iterable<A> && __backend::column_iterable<B> &&
           __detail::is_csc_view_v<C>)
void multiply_compute(operation_info_t& info, A&& a, B&& b, C&& c) {
  auto new_info = multiply_compute(std::forward<A>(a), std::forward<B>(b),
                                   std::forward<C>(c));
  info.update_impl_(new_info.result_shape(), new_info.result_nnz());
}

// C = AB
// SpGEMM fill stage with CSR or CSC output
template <matrix A, matrix B, matrix C>
void multiply_fill(operation_info_t info, A&& a, B&& b, C&& c) {
  log_trace("");
  multiply(a, b, c);
}

} // namespace spblas
