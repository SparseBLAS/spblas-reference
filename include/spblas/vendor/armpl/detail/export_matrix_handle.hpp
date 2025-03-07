#pragma once

#include <armpl_sparse.h>
#include <spblas/detail/view_inspectors.hpp>

namespace spblas {

namespace __armpl {

template <matrix M, typename O>
  requires __detail::is_csr_view_v<M>
void export_matrix_handle(operation_info_t& info, M&& m, armpl_spmat_t m_handle) {
  auto nnz = info.result_nnz();
  armpl_int_t m, n;
  armpl_int_t *rowptr, *colind;
  tensor_scalar_t<M>* values;
  __armpl::export_spmat_csr<tensor_scalar_t<M>>(m_handle, 0, &m, &n, &rowptr,
                                                &colind, &values);

  std::copy(values, values + nnz, m.values().begin());
  std::copy(colind, colind + nnz, m.colind().begin());
  std::copy(rowptr, rowptr + m + 1, m.rowptr().begin());

  free(values);
  free(rowptr);
  free(colind);
}

}

}