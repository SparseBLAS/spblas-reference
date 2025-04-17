#pragma once

#include <oneapi/mkl.hpp>

#include <spblas/detail/log.hpp>
#include <spblas/detail/operation_info_t.hpp>
#include <spblas/detail/ranges.hpp>
#include <spblas/detail/view_inspectors.hpp>
#include <spblas/views/matrix_opt.hpp>

namespace spblas {
namespace __mkl {

template <matrix A>
  requires __detail::has_csr_base<A>
oneapi::mkl::sparse::matrix_handle_t get_matrix_handle(sycl::queue& q,
                                                       A&& a_base) {
  oneapi::mkl::sparse::matrix_handle_t a_handle = nullptr;
  if constexpr (__detail::is_matrix_opt_view_v<decltype(a_base)>) {

    log_trace("using A as matrix_opt");
    a_handle = a_base.matrix_handle_;

    if (a_handle == nullptr) {
      oneapi::mkl::sparse::init_matrix_handle(&a_handle);

      auto a_csr = a_base.base();
      oneapi::mkl::sparse::set_csr_data(
          q, a_handle, __backend::shape(a_csr)[0], __backend::shape(a_csr)[1],
          oneapi::mkl::index_base::zero, a_csr.rowptr().data(),
          a_csr.colind().data(), a_csr.values().data())
          .wait();

      a_base.matrix_handle_ = a_handle;
    }
  } else {
    log_trace("using A as csr_base");

    oneapi::mkl::sparse::init_matrix_handle(&a_handle);

    oneapi::mkl::sparse::set_csr_data(
        q, a_handle, __backend::shape(a_base)[0], __backend::shape(a_base)[1],
        oneapi::mkl::index_base::zero, a_base.rowptr().data(),
        a_base.colind().data(), a_base.values().data())
        .wait();
  }

  return a_handle;
}

//
// potentially extract a_handle from info or a_base or build it
//
template <matrix A>
  requires __detail::has_csr_base<A>
oneapi::mkl::sparse::matrix_handle_t
get_matrix_handle(sycl::queue& q, A&& a_base,
                  oneapi::mkl::sparse::matrix_handle_t info_a_handle) {
  oneapi::mkl::sparse::matrix_handle_t a_handle = nullptr;

  if constexpr (__detail::is_matrix_opt_view_v<decltype(a_base)>) {

    log_trace("using A as matrix_opt");
    a_handle = a_base.matrix_handle_;

    if (a_handle == nullptr) {
      oneapi::mkl::sparse::init_matrix_handle(&a_handle);

      auto a_csr = a_base.base();
      oneapi::mkl::sparse::set_csr_data(
          q, a_handle, __backend::shape(a_csr)[0], __backend::shape(a_csr)[1],
          oneapi::mkl::index_base::zero, a_csr.rowptr().data(),
          a_csr.colind().data(), a_csr.values().data())
          .wait();

      a_base.matrix_handle_ = a_handle;
    }
  } else {

    if (info_a_handle != nullptr) {
      log_trace("using A from operation_info_t");
      return info_a_handle;
    } else {
      log_trace("using A as csr_base");

      oneapi::mkl::sparse::init_matrix_handle(&a_handle);

      oneapi::mkl::sparse::set_csr_data(
          q, a_handle, __backend::shape(a_base)[0], __backend::shape(a_base)[1],
          oneapi::mkl::index_base::zero, a_base.rowptr().data(),
          a_base.colind().data(), a_base.values().data())
          .wait();
    }

    return a_handle;
  }
}

} // namespace __mkl
} // namespace spblas
