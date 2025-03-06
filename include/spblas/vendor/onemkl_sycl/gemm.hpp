#pragma once

#include <oneapi/mkl.hpp>

#include <spblas/detail/ranges.hpp>
#include <spblas/detail/view_inspectors.hpp>

namespace spblas {

template <matrix A, matrix B, matrix C>
  requires __detail::has_mdspan_matrix_base<A> &&
           __detail::has_mdspan_matrix_base<B> &&
           __detail::is_matrix_instantiation_of_mdspan_v<C>
void multiply(A&& a, B&& b, C&& c) {
  auto a_base = __detail::get_ultimate_base(a);
  auto b_base = __detail::get_ultimate_base(a);

  auto alpha_optional = __detail::get_scaling_factor(a, b);
  tensor_scalar_t<A> alpha = alpha_optional.value_or(1);

  sycl::queue q(sycl::cpu_selector_v);

  /*
  namespace oneapi::mkl::blas::row_major {
    sycl::event gemm(sycl::queue &queue,
                     onemkl::transpose transa,
                     onemkl::transpose transb,
                     std::int64_t m,
                     std::int64_t n,
                     std::int64_t k,
                     Ts alpha,
                     const Ta *a,
                     std::int64_t lda,
                     const Tb *b,
                     std::int64_t ldb,
                     Ts beta,
                     Tc *c,
                     std::int64_t ldc,
                     const std::vector<sycl::event> &dependencies = {})
                     }
*/

  oneapi::mkl::blas::row_major::gemm(
      q, oneapi::mkl::transpose::nontrans, oneapi::mkl::transpose::nontrans,
      __backend::shape(a)[0], __backend::shape(c)[1], __backend::shape(a)[1],
      alpha, a_base.data_handle(), __backend::shape(a)[1], b_base.data_handle(),
      __backend::shape(b)[1], 0, c.data_handle(), __backend::shape(c)[1])
      .wait();
}

} // namespace spblas
