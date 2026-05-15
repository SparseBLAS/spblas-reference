#pragma once

#include <cusparse.h>

#include <stdexcept>

#include <spblas/detail/view_inspectors.hpp>

namespace spblas {

namespace __cusparse {

template <matrix M>
  requires __detail::is_csr_view_v<M>
cusparseSpMatDescr_t create_matrix_descriptor(M&& m) {
  cusparseSpMatDescr_t mat_descr;
  __cusparse::throw_if_error(cusparseCreateCsr(
      &mat_descr, __backend::shape(m)[0], __backend::shape(m)[1],
      m.values().size(), m.rowptr().data(), m.colind().data(),
      m.values().data(), detail::cusparse_index_type_v<tensor_offset_t<M>>,
      detail::cusparse_index_type_v<tensor_index_t<M>>,
      CUSPARSE_INDEX_BASE_ZERO, detail::cuda_data_type_v<tensor_scalar_t<M>>));

  return mat_descr;
}

template <matrix M>
  requires __detail::is_csc_view_v<M>
cusparseSpMatDescr_t create_matrix_descriptor(M&& m) {
  cusparseSpMatDescr_t mat_descr;
  __cusparse::throw_if_error(cusparseCreateCsc(
      &mat_descr, __backend::shape(m)[0], __backend::shape(m)[1],
      m.values().size(), m.rowptr().data(), m.colind().data(),
      m.values().data(), detail::cusparse_index_type_v<tensor_offset_t<M>>,
      detail::cusparse_index_type_v<tensor_index_t<M>>,
      CUSPARSE_INDEX_BASE_ZERO, detail::cuda_data_type_v<tensor_scalar_t<M>>));

  return mat_descr;
}

template <matrix M>
  requires __detail::has_base<M>
cusparseSpMatDescr_t create_matrix_descriptor(M&& m) {
  return create_matrix_descriptor(m.base());
}

} // namespace __cusparse

} // namespace spblas
