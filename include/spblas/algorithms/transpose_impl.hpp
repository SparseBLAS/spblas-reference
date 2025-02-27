#pragma once

#include "spblas/detail/view_inspectors.hpp"
#include <spblas/concepts.hpp>
#include <spblas/detail/operation_info_t.hpp>

namespace spblas {

template <matrix A, matrix B>
operation_info_t transpose_inspect(A&& a, B&& b) {
  return {};
}

template <matrix A, matrix B>
  requires(__detail::is_csr_view_v<A> && __detail::is_csr_view_v<B>)
void transpose(operation_info_t& info, A&& a, B&& b) {
  if (__backend::shape(a)[0] != __backend::shape(b)[1] ||
      __backend::shape(a)[1] != __backend::shape(b)[0]) {
    throw std::invalid_argument(
        "transpose: matrix dimensions are incompatible.");
  }
  if (__backend::size(a) != __backend::size(b)) {
    throw std::invalid_argument("transpose: matrix nnz are incompatible.");
  }
  using O = tensor_offset_t<B>;

  const auto b_base = __detail::get_ultimate_base(b);
  const auto b_rowptr = b_base.rowptr();
  const auto b_colind = b_base.colind();
  const auto b_values = b_base.values();

  __ranges::fill(b_rowptr, 0);

  for (auto&& [i, row] : __backend::rows(a)) {
    for (auto&& [j, _] : row) {
      b_rowptr[j + 1]++;
    }
  }

  std::exclusive_scan(b_rowptr.begin(), b_rowptr.end(), b_rowptr.begin(), O{});

  for (auto&& [i, row] : __backend::rows(a)) {
    for (auto&& [j, v] : row) {
      const auto out_idx = b_rowptr[j + 1];
      b_colind[out_idx] = i;
      b_values[out_idx] = v;
      b_rowptr[j + 1]++;
    }
  }
}

} // namespace spblas
