#pragma once

#include <spblas/detail/view_inspectors.hpp>

namespace spblas {

template <matrix M>
  requires(__detail::is_csr_view_v<M>)
auto transposed(M&& m) {
  return csc_view<tensor_scalar_t<M>, tensor_index_t<M>, tensor_offset_t<M>>(
      m.values(), m.rowptr(), m.colind(), {m.shape()[1], m.shape()[0]},
      m.size());
}

template <matrix M>
  requires(__detail::is_csc_view_v<M>)
auto transposed(M&& m) {
  return csr_view<tensor_scalar_t<M>, tensor_index_t<M>, tensor_offset_t<M>>(
      m.values(), m.colptr(), m.rowind(), {m.shape()[1], m.shape()[0]},
      m.size());
}

} // namespace spblas
