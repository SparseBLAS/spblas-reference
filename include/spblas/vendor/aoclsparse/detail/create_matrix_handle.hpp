#pragma once

#include <spblas/vendor/aoclsparse/aocl_wrappers.hpp>

#include <spblas/detail/view_inspectors.hpp>

namespace spblas {

namespace __aoclsparse {

template <matrix M>
  requires __detail::is_csr_view_v<M>
aoclsparse_matrix create_matrix_handle(M&& m) {
  aoclsparse_matrix handle = nullptr;
  aoclsparse_status status = __aoclsparse::aoclsparse_create_csr(
      &handle, aoclsparse_index_base_zero, m.shape()[0], m.shape()[1], m.size(),
      m.rowptr().data(), m.colind().data(), m.values().data());

  if (status != aoclsparse_status_success) {
    throw std::runtime_error("create_matrix_handle: AOCL-Sparse failed while "
                             "creating matrix handle.");
  }

  return handle;
}

template <matrix M>
  requires __detail::is_csc_view_v<M>
aoclsparse_matrix create_matrix_handle(M&& m) {
  aoclsparse_matrix handle = nullptr;
  aoclsparse_status status = __aoclsparse::aoclsparse_create_csr(
      &handle, aoclsparse_index_base_zero, m.shape()[1], m.shape()[0], m.size(),
      m.colptr().data(), m.rowind().data(), m.values().data());

  if (status != aoclsparse_status_success) {
    throw std::runtime_error("create_matrix_handle: AOCL-Sparse failed while "
                             "creating matrix handle.");
  }

  return handle;
}

template <matrix M>
aoclsparse_operation get_transpose(M&& m) {
  static_assert(__detail::has_csr_base<M> || __detail::has_csc_base<M>);
  if constexpr (__detail::has_base<M>) {
    return get_transpose(m.base());
  } else if constexpr (__detail::is_csr_view_v<M>) {
    return aoclsparse_operation_none;
  } else if constexpr (__detail::is_csc_view_v<M>) {
    return aoclsparse_operation_transpose;
  }
}

} // namespace __aoclsparse

} // namespace spblas
