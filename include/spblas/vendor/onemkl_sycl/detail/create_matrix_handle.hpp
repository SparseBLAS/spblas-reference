#pragma once

#include <oneapi/mkl.hpp>

#include <spblas/detail/view_inspectors.hpp>

namespace spblas {

namespace __mkl {

template <matrix M>
  requires __detail::is_csr_view_v<M>
oneapi::mkl::sparse::matrix_handle_t create_matrix_handle(sycl::queue& q,
                                                          M&& m) {
  oneapi::mkl::sparse::matrix_handle_t handle = nullptr;
  oneapi::mkl::sparse::init_matrix_handle(&handle);

  oneapi::mkl::sparse::set_csr_data(
      q, handle, m.shape()[0], m.shape()[1], oneapi::mkl::index_base::zero,
      m.rowptr().data(), m.colind().data(), m.values().data())
      .wait();

  return handle;
}

template <matrix M>
  requires __detail::is_csc_view_v<M>
oneapi::mkl::sparse::matrix_handle_t create_matrix_handle(sycl::queue& q,
                                                          M&& m) {
  oneapi::mkl::sparse::matrix_handle_t handle = nullptr;
  oneapi::mkl::sparse::init_matrix_handle(&handle);

  oneapi::mkl::sparse::set_csr_data(
      q, handle, m.shape()[1], m.shape()[0], oneapi::mkl::index_base::zero,
      m.colptr().data(), m.rowind().data(), m.values().data())
      .wait();

  return handle;
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
template <matrix M>
oneapi::mkl::transpose get_transpose(M&& m) {
  static_assert(__detail::has_base<M> || __detail::is_csr_view_v<M> ||
                __detail::is_csc_view_v<M>);
  if constexpr (__detail::has_base<M>) {
    return get_transpose(m.base());
  } else if constexpr (__detail::is_csr_view_v<M>) {
    return oneapi::mkl::transpose::nontrans;
  } else if constexpr (__detail::is_csc_view_v<M>) {
    return oneapi::mkl::transpose::trans;
  }
}

} // namespace __mkl

} // namespace spblas
