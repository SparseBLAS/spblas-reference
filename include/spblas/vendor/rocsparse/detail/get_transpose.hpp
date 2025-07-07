#pragma once

#include <rocsparse/rocsparse.h>
#include <spblas/detail/view_inspectors.hpp>

namespace spblas {
namespace __rocsparse {

//
// Takes in a CSR or CSR_transpose (aka CSC) or CSC or CSC_transpose
// and returns the rocsparse_operation value associated with it being
// represented in the CSR format
//
//     CSR = CSR + NON_TRANSPOSE
//     CSR_transpose = CSR + TRANSPOSE
//     CSC = CSR + TRANSPOSE
//     CSC_transpose = CSR + NON_TRANSPOSE
//
template <matrix M>
rocsparse_operation get_transpose(M&& m) {
  static_assert(__detail::has_csr_base<M> || __detail::has_csc_base<M>);
  if constexpr (__detail::has_base<M>) {
    return get_transpose(m.base());
  } else if constexpr (__detail::is_csr_view_v<M>) {
    return rocsparse_operation_none;
  } else if constexpr (__detail::is_csc_view_v<M>) {
    return rocsparse_operation_transpose;
  }
}

} // namespace __rocsparse
} // namespace spblas
