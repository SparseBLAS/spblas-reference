#pragma once

#include <oneapi/mkl.hpp>

#include <fmt/ranges.h>
#include <spblas/detail/ranges.hpp>

namespace spblas {

template <matrix A, vector B, vector C>
  requires __detail::is_csr_view_v<A> && __ranges::contiguous_range<B> &&
           __ranges::contiguous_range<C>
void multiply(A&& a, B&& b, C&& c) {
  fmt::print("Hello from oneMKL!!\n");
  sycl::queue q(sycl::cpu_selector_v);
  oneapi::mkl::sparse::matrix_handle_t a_handle = nullptr;

  oneapi::mkl::sparse::init_matrix_handle(&a_handle);

  oneapi::mkl::sparse::set_csr_data(
      q, a_handle, __backend::shape(a)[0], __backend::shape(a)[1],
      oneapi::mkl::index_base::zero, a.rowptr().data(), a.colind().data(),
      a.values().data())
      .wait();

  oneapi::mkl::sparse::optimize_gemv(q, oneapi::mkl::transpose::nontrans,
                                     a_handle)
      .wait();

  oneapi::mkl::sparse::gemv(q, oneapi::mkl::transpose::nontrans, 1.0, a_handle,
                            __ranges::data(b), 0.0, __ranges::data(c))
      .wait();

  oneapi::mkl::sparse::release_matrix_handle(q, &a_handle).wait();
}

} // namespace spblas
