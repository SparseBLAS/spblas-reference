#pragma once

#include <type_traits>

#include <cusparse.h>

#include <spblas/concepts.hpp>
#include <spblas/views/inspectors.hpp>

#include "exception.hpp"
#include "types.hpp"

namespace spblas {
namespace __cusparse {

// create matrix descriptor from spblas csr view
template <matrix mat>
  requires __detail::is_csr_view_v<mat>
cusparseSpMatDescr_t create_matrix_descr(mat&& a) {
  using matrix_type = std::remove_cvref_t<mat>;
  cusparseSpMatDescr_t descr;
  throw_if_error(cusparseCreateCsr(
      &descr, __backend::shape(a)[0], __backend::shape(a)[1], a.values().size(),
      a.rowptr().data(), a.colind().data(), a.values().data(),
      to_cusparse_indextype<typename matrix_type::offset_type>(),
      to_cusparse_indextype<typename matrix_type::index_type>(),
      CUSPARSE_INDEX_BASE_ZERO,
      to_cuda_datatype<typename matrix_type::scalar_type>()));
  return descr;
}

// create dense vector from mdspan
template <vector vec>
  requires __ranges::contiguous_range<vec>
cusparseDnVecDescr_t create_vector_descr(vec&& v) {
  using vector_type = std::remove_cvref_t<vec>;
  cusparseDnVecDescr_t descr;
  throw_if_error(cusparseCreateDnVec(
      &descr, v.size(), v.data(),
      to_cuda_datatype<typename vector_type::value_type>()));
  return descr;
}

} // namespace __cusparse
} // namespace spblas
