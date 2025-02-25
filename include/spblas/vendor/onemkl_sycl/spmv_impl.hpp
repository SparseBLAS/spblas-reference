#pragma once

#include <oneapi/mkl.hpp>

#include <spblas/detail/log.hpp>
#include <spblas/detail/operation_info_t.hpp>
#include <spblas/detail/ranges.hpp>
#include <spblas/detail/view_inspectors.hpp>

//
// Defines the following APIs for SpMV:
//
// y = alpha* op(A) * x
//
//  where A is a sparse matrices of CSR format and
//  x/y are dense vectors
//
// //operation_info_t multiply_inspect(A, x, y)
// //void multiply_inspect(operation_info_t, A, x, y)
//
// //void multiply_execute(operation_info_t, A, x, y)
// void multiply(A, x, y)
//

namespace spblas {

template <matrix A, vector X, vector Y>
  requires __detail::has_csr_base<A> &&
           __detail::has_contiguous_range_base<X> &&
           __ranges::contiguous_range<Y>
void multiply(A&& a, X&& x, Y&& y) {
  log_trace("");
  auto a_base = __detail::get_ultimate_base(a);
  auto x_base = __detail::get_ultimate_base(x);

  auto alpha_optional = __detail::get_scaling_factor(a, x);
  tensor_scalar_t<A> alpha = alpha_optional.value_or(1);

  sycl::queue q(sycl::cpu_selector_v);
  oneapi::mkl::sparse::matrix_handle_t a_handle = nullptr;

  oneapi::mkl::sparse::init_matrix_handle(&a_handle);

  oneapi::mkl::sparse::set_csr_data(
      q, a_handle, __backend::shape(a_base)[0], __backend::shape(a_base)[1],
      oneapi::mkl::index_base::zero, a_base.rowptr().data(),
      a_base.colind().data(), a_base.values().data())
      .wait();

  oneapi::mkl::sparse::gemv(q, oneapi::mkl::transpose::nontrans, alpha,
                            a_handle, __ranges::data(x_base), 0.0,
                            __ranges::data(y))
      .wait();

  oneapi::mkl::sparse::release_matrix_handle(q, &a_handle).wait();
}

} // namespace spblas
