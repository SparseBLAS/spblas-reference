#pragma once

#include <cusparse.h>

#include <stdexcept>

#include <spblas/detail/view_inspectors.hpp>

namespace spblas {

namespace __cusparse {

template <matrix M>
  requires __detail::is_csr_view_v<M>
cusparseSpMatDescr_t create_matrix_handle(M&& m) {
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
cusparseSpMatDescr_t create_matrix_handle(M&& m) {
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
cusparseSpMatDescr_t create_matrix_handle(M&& m) {
  return create_matrix_handle(m.base());
}

//
// Takes in a CSR or CSR_transpose (aka CSC) or CSC or CSC_transpose
//  and returns the transpose value associated with it being represented
// in the CSR format (since oneMKL SYCL currently does not have CSC
// format
//
//     CSR = CSR + nontrans
//     CSR_transpose = CSR + trans
//     CSC = CSR + trans
//     CSC_transpose -> CSR + nontrans
//
// template <matrix M>
// oneapi::mkl::transpose get_transpose(M&& m) {
//   static_assert(__detail::has_csr_base<M> || __detail::has_csc_base<M>);

//   const bool conjugate = __detail::is_conjugated(m);
//   if constexpr (__detail::has_csr_base<M>) {
//     if (conjugate) {
//       throw std::runtime_error(
//           "oneMKL SYCL backend does not support conjugation for CSR views.");
//     }
//     return oneapi::mkl::transpose::nontrans;
//   } else if constexpr (__detail::has_csc_base<M>) {
//     return conjugate ? oneapi::mkl::transpose::conjtrans
//                      : oneapi::mkl::transpose::trans;
//   }
// }

} // namespace __cusparse

} // namespace spblas
