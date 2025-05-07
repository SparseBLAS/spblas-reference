#pragma once

#include <type_traits>

#include <rocsparse/rocsparse.h>

#include <spblas/concepts.hpp>
#include <spblas/views/inspectors.hpp>

#include "exception.hpp"
#include "types.hpp"

namespace spblas {
namespace __rocsparse {

// create matrix descriptor from spblas csr view
template <matrix mat>
  requires __detail::is_csr_view_v<mat>
rocsparse_spmat_descr create_matrix_descr(mat&& a) {
  using matrix_type = std::remove_cvref_t<mat>;
  rocsparse_spmat_descr descr;
  throw_if_error(rocsparse_create_csr_descr(
      &descr, __backend::shape(a)[0], __backend::shape(a)[1], a.values().size(),
      a.rowptr().data(), a.colind().data(), a.values().data(),
      to_rocsparse_indextype<typename matrix_type::offset_type>(),
      to_rocsparse_indextype<typename matrix_type::index_type>(),
      rocsparse_index_base_zero,
      to_rocsparse_datatype<typename matrix_type::scalar_type>()));
  return descr;
}

} // namespace __rocsparse
} // namespace spblas
