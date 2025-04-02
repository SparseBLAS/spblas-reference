#pragma once

#include <oneapi/mkl.hpp>

#include "mkl_allocator.hpp"
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
// //void multiply_compute(operation_info_t, A, x, y)
// void multiply(A, x, y)
//

namespace spblas {

class spmv_state_t {
public:
  spmv_state_t() : spmv_state_t(mkl::mkl_allocator<char>{}) {}

  spmv_state_t(sycl::queue* q) : spmv_state_t(mkl::mkl_allocator<char>{q}) {}

  spmv_state_t(mkl::mkl_allocator<char> alloc) : alloc_(alloc) {}

  sycl::queue* queue() {
    return alloc_.queue();
  }

private:
  mkl::mkl_allocator<char> alloc_;
};

template <matrix A, vector X, vector Y>
  requires((__detail::has_csr_base<A> || __detail::has_csc_base<A>) &&
           __detail::has_contiguous_range_base<X> &&
           __ranges::contiguous_range<Y>)
void multiply(spmv_state_t& state, A&& a, X&& x, Y&& y) {
  log_trace("");
  auto a_base = __detail::get_ultimate_base(a);
  auto x_base = __detail::get_ultimate_base(x);

  auto alpha_optional = __detail::get_scaling_factor(a, x);
  tensor_scalar_t<A> alpha = alpha_optional.value_or(1);

  auto q_ptr = state.queue();

  auto a_handle = __mkl::create_matrix_handle(*q_ptr, a_base);
  auto a_transpose = __mkl::get_transpose(a);

  oneapi::mkl::sparse::gemv(*q_ptr, a_transpose, alpha, a_handle,
                            __ranges::data(x_base), 0.0, __ranges::data(y))
      .wait();

  oneapi::mkl::sparse::release_matrix_handle(*q_ptr, &a_handle).wait();
}

template <matrix A, vector X, vector Y>
  requires((__detail::has_csr_base<A> || __detail::has_csc_base<A>) &&
           __detail::has_contiguous_range_base<X> &&
           __ranges::contiguous_range<Y>)
void multiply(A&& a, X&& x, Y&& y) {
  spmv_state_t state;
  multiply(state, a, x, y);
}

} // namespace spblas
