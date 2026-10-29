#pragma once

#include <utility>

#include <spblas/detail/view_inspectors.hpp>

namespace spblas {

namespace __detail {

// A CSR matrix's arrays as the CSC arrays of its transpose, and vice versa:
// what view::transpose returns for a wrapper.
template <matrix M>
  requires(is_csr_view_v<M>)
auto reinterpret_transposed(M&& m) {
  return csc_view<tensor_scalar_t<M>, tensor_index_t<M>, tensor_offset_t<M>>(
      m.values(), m.rowptr(), m.colind(), {m.shape()[1], m.shape()[0]},
      m.size());
}

template <matrix M>
  requires(is_csc_view_v<M>)
auto reinterpret_transposed(M&& m) {
  return csr_view<tensor_scalar_t<M>, tensor_index_t<M>, tensor_offset_t<M>>(
      m.values(), m.colptr(), m.rowind(), {m.shape()[1], m.shape()[0]},
      m.size());
}

} // namespace __detail

// Deprecated: use view::transpose. Kept until every backend has migrated.
template <matrix M>
  requires(__detail::is_csr_view_v<M> || __detail::is_csc_view_v<M>)
auto transposed(M&& m) {
  return __detail::reinterpret_transposed(std::forward<M>(m));
}

} // namespace spblas
