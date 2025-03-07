#pragma once

#include <armpl_sparse.h>
#include <spblas/detail/view_inspectors.hpp>

namespace spblas {

namespace __armpl {

template <matrix M>
  requires __detail::is_csr_view_v<M>
armpl_spmat_t create_matrix_handle(M&& m) {
  armpl_spmat_t handle;
  __armpl::create_spmat_csr<tensor_scalar_t<M>>(
      &handle, m.shape()[0], m.shape()[1], m.rowptr().data(), m.colind().data(),
      m.values().data(), ARMPL_SPARSE_CREATE_NOCOPY);
  return handle;
}

template <matrix M>
  requires __detail::is_csc_view_v<M>
armpl_spmat_t create_matrix_handle(M&& m) {
  armpl_spmat_t handle;
  __armpl::create_spmat_csc<tensor_scalar_t<M>>(
      &handle, m.shape()[0], m.shape()[1], m.rowind().data(), m.colptr().data(),
      m.values().data(), ARMPL_SPARSE_CREATE_NOCOPY);
  return handle;
}

} // namespace __armpl

} // namespace spblas
