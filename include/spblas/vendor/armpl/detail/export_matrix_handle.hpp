#pragma once

#include <spblas/detail/operation_info_t.hpp>

#include <armpl_sparse.h>
#include <spblas/detail/view_inspectors.hpp>

namespace spblas {

namespace __armpl {

template <matrix M>
  requires __detail::is_csr_view_v<M>
void export_matrix_handle(operation_info_t& info, M&& matrix,
                          armpl_spmat_t matrix_handle) {
  auto nnz = info.result_nnz();
  armpl_int_t m, n;
  armpl_int_t *rowptr, *colind;
  tensor_scalar_t<M>* values;
  __armpl::export_spmat_csr<tensor_scalar_t<M>>(matrix_handle, 0, &m, &n,
                                                &rowptr, &colind, &values);

  std::copy(values, values + nnz, matrix.values().begin());
  std::copy(colind, colind + nnz, matrix.colind().begin());
  std::copy(rowptr, rowptr + m + 1, matrix.rowptr().begin());

  free(values);
  free(rowptr);
  free(colind);
}

template <matrix M>
  requires __detail::is_csc_view_v<M>
void export_matrix_handle(operation_info_t& info, M&& matrix,
                          armpl_spmat_t matrix_handle) {
  auto nnz = info.result_nnz();
  armpl_int_t m, n;
  armpl_int_t *colptr, *rowind;
  tensor_scalar_t<M>* values;
  __armpl::export_spmat_csc<tensor_scalar_t<M>>(matrix_handle, 0, &m, &n,
                                                &rowind, &colptr, &values);

  std::copy(values, values + nnz, matrix.values().begin());
  std::copy(rowind, rowind + nnz, matrix.rowind().begin());
  std::copy(colptr, colptr + n + 1, matrix.colptr().begin());

  free(values);
  free(colptr);
  free(rowind);
}

} // namespace __armpl

} // namespace spblas
