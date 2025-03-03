#pragma once

#include <oneapi/mkl.hpp>

#include <spblas/detail/log.hpp>
#include <spblas/detail/operation_info_t.hpp>
#include <spblas/detail/ranges.hpp>
#include <spblas/detail/view_inspectors.hpp>
#include <spblas/views/matrix_opt.hpp>
#include "matrix_wrapper.hpp"

//
// Defines the following APIs for SpMM:
//
//  Y = alpha * op(A) * X
//
//  where A is a sparse matrices of CSR format and
//  X/Y are dense matrices of row_major format
//
// //operation_info_t multiply_inspect(A, x, y)
// //void multiply_inspect(operation_info_t, A, x, y)
//
// //void multiply_compute(operation_info_t, A, x, y)
// void multiply(A, x, y)
//

namespace spblas {

template <matrix A, matrix X, matrix Y>
  requires __detail::has_csr_base<A> && __detail::has_mdspan_matrix_base<X> &&
           __detail::is_matrix_instantiation_of_mdspan_v<Y> &&
           std::is_same_v<
               typename __detail::ultimate_base_type_t<X>::layout_type,
               __mdspan::layout_right> &&
           std::is_same_v<typename std::remove_cvref_t<Y>::layout_type,
                          __mdspan::layout_right>
void multiply(A&& a, X&& x, Y&& y) {
  log_trace("");
  auto a_base = __detail::get_ultimate_base(a);
  auto x_base = __detail::get_ultimate_base(x);

  auto alpha_optional = __detail::get_scaling_factor(a, x);
  tensor_scalar_t<A> alpha = alpha_optional.value_or(1);

  sycl::queue q(sycl::cpu_selector_v);

  auto a_handle = __mkl::get_matrix_handle(q, a_base);
 
  oneapi::mkl::sparse::gemm(
          q, oneapi::mkl::layout::row_major, oneapi::mkl::transpose::nontrans,
          oneapi::mkl::transpose::nontrans, alpha, a_handle, x_base.data_handle(),
          x_base.extent(1), x_base.extent(1), 0.0, y.data_handle(), y.extent(1))
      .wait();


  if constexpr (!__detail::is_matrix_opt_view_v<decltype(a_base)>) {
      oneapi::mkl::sparse::release_matrix_handle(q, &a_handle).wait();
  }
}

} // namespace spblas
