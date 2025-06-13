#pragma once

#include <spblas/backend/backend.hpp>
#include <spblas/concepts.hpp>
#include <spblas/detail/log.hpp>

#include <spblas/algorithms/transposed.hpp>
#include <spblas/backend/csr_builder.hpp>
#include <spblas/backend/spa_accumulator.hpp>
#include <spblas/detail/operation_info_t.hpp>
#include <spblas/detail/view_inspectors.hpp>

#include "detail/spgemm/spgemm.hpp"

#include <algorithm>

namespace spblas {

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

  __backend::for_each(c, [](auto&& e) {
    auto&& [_, v] = e;
    v = 0;
  });

  __backend::for_each(a, [&](auto&& e) {
    auto&& [idx, a_v] = e;
    auto&& [i, k] = idx;
    for (std::size_t j = 0; j < __backend::shape(b)[1]; j++) {
      __backend::lookup(c, i, j) += a_v * __backend::lookup(b, k, j);
    }
  });
}

template <matrix A, matrix B, matrix C>
operation_info_t multiply_inspect(A&& a, B&& b, C&& c) {
  return operation_info_t{};
}

template <matrix A, matrix B, matrix C>
void multiply_inspect(operation_info_t& info, A&& a, B&& b, C&& c){};

template <matrix A, matrix B, matrix C>
  requires(__backend::row_iterable<A> && __backend::row_iterable<B> &&
           __detail::is_csr_view_v<C>)
void multiply_compute(operation_info_t& info, A&& a, B&& b, C&& c) {
  auto new_info = multiply_compute(std::forward<A>(a), std::forward<B>(b),
                                   std::forward<C>(c));
  info.update_impl_(new_info.result_shape(), new_info.result_nnz());
}

// C = AB
// SpGEMM (Gustavson's Algorithm) on existing C values
template <matrix A, matrix B, matrix C>
  requires(__backend::row_iterable<A> && __backend::row_iterable<B> &&
           __detail::is_csr_view_v<C>)
void multiply_update(A&& a, B&& b, C&& c) {
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

  auto c_base = __detail::get_ultimate_base(c);
  const auto c_rowptr = c_base.rowptr();
  const auto c_colind = c_base.colind();
  const auto c_values = c_base.values();

  for (auto&& [i, a_row] : __backend::rows(a)) {
    std::unordered_map<I, O> c_columns;
    const auto c_begin = c_rowptr[i];
    const auto c_end = c_rowptr[i + 1];
    for (auto c_nz : __ranges::views::iota(c_begin, c_end)) {
      c_columns.emplace(c_colind[c_nz], c_nz);
      c_values[c_nz] = 0;
    }
    for (auto&& [k, a_v] : a_row) {
      for (auto&& [j, b_v] : __backend::lookup_row(b, k)) {
        c_values[c_columns[j]] += a_v * b_v;
      }
    }
  }
}

template <matrix A, matrix B, matrix C>
  requires(__backend::column_iterable<A> && __backend::column_iterable<B> &&
           __detail::is_csc_view_v<C>)
void multiply_compute(operation_info_t& info, A&& a, B&& b, C&& c) {
  auto new_info = multiply_compute(std::forward<A>(a), std::forward<B>(b),
                                   std::forward<C>(c));
  info.update_impl_(new_info.result_shape(), new_info.result_nnz());
}

// C = AB
template <matrix A, matrix B, matrix C>
void multiply_fill(operation_info_t info, A&& a, B&& b, C&& c) {
  log_trace("");
  multiply(a, b, c);
}

// C = AB after multiply_fill(info, A, B, C) was called previously
template <matrix A, matrix B, matrix C>
void multiply_fill_update(operation_info_t info, A&& a, B&& b, C&& c) {
  log_trace("");
  multiply_update(a, b, c);
}

} // namespace spblas
