#pragma once

#include <rocsparse/rocsparse.h>

#include <spblas/detail/types.hpp>
#include <spblas/detail/view_inspectors.hpp>
#include <spblas/vendor/rocsparse/exception.hpp>
#include <spblas/vendor/rocsparse/types.hpp>

namespace spblas {
namespace __rocsparse {

template <matrix M>
  requires __detail::is_csr_view_v<M>
rocsparse_spmat_descr create_rocsparse_handle(M&& m) {
  rocsparse_spmat_descr mat_descr;
  throw_if_error(rocsparse_create_csr_descr(
      &mat_descr, __backend::shape(m)[0], __backend::shape(m)[1],
      m.values().size(), m.rowptr().data(), m.colind().data(),
      m.values().data(), detail::rocsparse_index_type_v<tensor_offset_t<M>>,
      detail::rocsparse_index_type_v<tensor_index_t<M>>,
      rocsparse_index_base_zero,
      detail::rocsparse_data_type_v<tensor_scalar_t<M>>));

  return mat_descr;
}

template <vector V>
  requires __ranges::contiguous_range<V>
rocsparse_dnvec_descr create_rocsparse_handle(V&& v) {
  rocsparse_dnvec_descr vec_descr;
  throw_if_error(rocsparse_create_dnvec_descr(
      &vec_descr, __backend::shape(v), __ranges::data(v),
      detail::rocsparse_data_type_v<tensor_scalar_t<V>>));

  return vec_descr;
}

} // namespace __rocsparse
} // namespace spblas
