#pragma once

#include <cusparse.h>

#include <spblas/detail/types.hpp>
#include <spblas/detail/view_inspectors.hpp>
#include <spblas/vendor/cusparse/types.hpp>

namespace spblas {

namespace __cusparse {

template <matrix M>
  requires __detail::is_csr_view_v<M>
cusparseSpMatDescr_t create_cusparse_handle(M&& m) {
  cusparseSpMatDescr_t mat_descr;
  __cusparse::throw_if_error(cusparseCreateCsr(
      &mat_descr, __backend::shape(m)[0], __backend::shape(m)[1],
      m.values().size(), m.rowptr().data(), m.colind().data(),
      m.values().data(), detail::cusparse_index_type_v<tensor_offset_t<M>>,
      detail::cusparse_index_type_v<tensor_index_t<M>>,
      CUSPARSE_INDEX_BASE_ZERO, detail::cuda_data_type_v<tensor_scalar_t<M>>));

  return mat_descr;
}

template <vector V>
  requires __ranges::contiguous_range<V>
cusparseDnVecDescr_t create_cusparse_handle(V&& v) {
  cusparseDnVecDescr_t vec_descr;
  __cusparse::throw_if_error(
      cusparseCreateDnVec(&vec_descr, __backend::shape(v), __ranges::data(v),
                          detail::cuda_data_type_v<tensor_scalar_t<V>>));

  return vec_descr;
}

} // namespace __cusparse

} // namespace spblas
