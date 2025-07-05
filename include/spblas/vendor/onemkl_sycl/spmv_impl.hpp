#pragma once

#include <oneapi/mkl.hpp>

#include <spblas/detail/log.hpp>
#include <spblas/detail/operation_info_t.hpp>
#include <spblas/detail/ranges.hpp>
#include <spblas/detail/view_inspectors.hpp>

#include <spblas/vendor/onemkl_sycl/detail/detail.hpp>

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
// //void multiply_compute(operation_info_t, A, x, y)
// void multiply(A, x, y)
//

namespace spblas {

template <typename ExecutionPolicy, matrix A, vector X, vector Y>
  requires((__detail::has_csr_base<A> || __detail::has_csc_base<A>) &&
           __detail::has_contiguous_range_base<X> &&
           __ranges::contiguous_range<Y>)
void multiply(ExecutionPolicy&& policy, A&& a, X&& x, Y&& y) {
  log_trace("");
  auto x_base = __detail::get_ultimate_base(x);

  auto alpha_optional = __detail::get_scaling_factor(a, x);
  tensor_scalar_t<A> alpha = alpha_optional.value_or(1);

  auto a_data = __detail::get_ultimate_base(a).values().data();

  auto&& q = __mkl::get_queue(policy, a_data);

  auto a_handle = __mkl::get_matrix_handle(q, a);
  auto a_transpose = __mkl::get_transpose(a);

  oneapi::mkl::sparse::gemv(q, a_transpose, alpha, a_handle,
                            __ranges::data(x_base), 0.0, __ranges::data(y))
      .wait();

  if (!__detail::has_matrix_opt(a)) {
    oneapi::mkl::sparse::release_matrix_handle(q, &a_handle).wait();
  }
}

template <matrix A, vector X, vector Y>
  requires((__detail::has_csr_base<A> || __detail::has_csc_base<A>) &&
           __detail::has_contiguous_range_base<X> &&
           __ranges::contiguous_range<Y>)
void multiply(A&& a, X&& x, Y&& y) {
  multiply(mkl::par, std::forward<A>(a), std::forward<X>(x),
           std::forward<Y>(y));
}

} // namespace spblas
