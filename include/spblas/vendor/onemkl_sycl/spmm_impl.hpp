#pragma once

#include <oneapi/mkl.hpp>

#include <stdexcept>

#include <spblas/detail/log.hpp>
#include <spblas/detail/operation_info_t.hpp>
#include <spblas/detail/ranges.hpp>
#include <spblas/detail/view_inspectors.hpp>
#include <spblas/views/matrix_opt.hpp>

#include <spblas/vendor/onemkl_sycl/detail/detail.hpp>

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

template <typename ExecutionPolicy, matrix A, matrix X, matrix Y>
  requires(
      (__detail::has_csr_base<A> || __detail::has_csc_base<A>) &&
      __detail::has_mdspan_matrix_base<X> && __detail::is_matrix_mdspan_v<Y> &&
      std::is_same_v<typename __detail::ultimate_base_type_t<X>::layout_type,
                     __mdspan::layout_right> &&
      std::is_same_v<typename std::remove_cvref_t<Y>::layout_type,
                     __mdspan::layout_right>)
void multiply_inspect(ExecutionPolicy&& policy, operation_info_t& info, A&& a,
                      X&& x, Y&& y) {
  log_trace("");
  if (__detail::is_conjugated(x) || __detail::is_conjugated(y)) {
    throw std::runtime_error(
        "oneMKL SYCL backend does not support conjugated dense matrices.");
  }

  if (__detail::has_matrix_opt(a)) {
    auto a_data = __detail::get_ultimate_base(a).values().data();
    auto&& q = __mkl::get_queue(policy, a_data);

    auto a_handle = __mkl::get_matrix_handle(q, a);
    auto a_transpose = __mkl::get_transpose(a);

    auto x_base = __detail::get_ultimate_base(x);

    oneapi::mkl::sparse::optimize_gemm(
        q, oneapi::mkl::layout::row_major, a_transpose,
        oneapi::mkl::transpose::nontrans, a_handle,
        static_cast<std::int64_t>(x_base.extent(1)))
        .wait();
  } else {
    // do nothing, since it would be immediately discarded
    log_info(
        "No work done, since no matrix_opt to store optimized results into!");
  }
} // multiply_inspect

template <typename ExecutionPolicy, matrix A, matrix X, matrix Y>
  requires(
      (__detail::has_csr_base<A> || __detail::has_csc_base<A>) &&
      __detail::has_mdspan_matrix_base<X> && __detail::is_matrix_mdspan_v<Y> &&
      std::is_same_v<typename __detail::ultimate_base_type_t<X>::layout_type,
                     __mdspan::layout_right> &&
      std::is_same_v<typename std::remove_cvref_t<Y>::layout_type,
                     __mdspan::layout_right>)
operation_info_t multiply_inspect(ExecutionPolicy&& policy, A&& a, X&& x,
                                  Y&& y) {
  log_trace("");
  operation_info_t info{};

  multiply_inspect(std::forward<ExecutionPolicy>(policy), info,
                   std::forward<A>(a), std::forward<X>(x), std::forward<Y>(y));

  return info;
}

template <typename ExecutionPolicy, matrix A, matrix X, matrix Y>
  requires(
      (__detail::has_csr_base<A> || __detail::has_csc_base<A>) &&
      __detail::has_mdspan_matrix_base<X> && __detail::is_matrix_mdspan_v<Y> &&
      std::is_same_v<typename __detail::ultimate_base_type_t<X>::layout_type,
                     __mdspan::layout_right> &&
      std::is_same_v<typename std::remove_cvref_t<Y>::layout_type,
                     __mdspan::layout_right>)
void multiply(ExecutionPolicy&& policy, operation_info_t& info, A&& a, X&& x,
              Y&& y) {
  log_trace("");

  if (__detail::is_conjugated(x) || __detail::is_conjugated(y)) {
    throw std::runtime_error(
        "oneMKL SYCL backend does not support conjugated dense matrices.");
  }

  auto alpha_optional = __detail::get_scaling_factor(a, x);
  tensor_scalar_t<A> alpha = alpha_optional.value_or(1);

  auto a_data = __detail::get_ultimate_base(a).values().data();
  auto&& q = __mkl::get_queue(policy, a_data);

  auto a_handle = __mkl::get_matrix_handle(q, a);
  auto a_transpose = __mkl::get_transpose(a);

  auto x_base = __detail::get_ultimate_base(x);

  oneapi::mkl::sparse::gemm(q, oneapi::mkl::layout::row_major, a_transpose,
                            oneapi::mkl::transpose::nontrans, alpha, a_handle,
                            x_base.data_handle(), x_base.extent(1),
                            x_base.extent(1), 0.0, y.data_handle(), y.extent(1))
      .wait();

  if (!__detail::has_matrix_opt(a)) {
    oneapi::mkl::sparse::release_matrix_handle(q, &a_handle).wait();
  }
}

//
// multiply_inspect - CSR/CSC with row major dense matrix rhs without execution
// policy
//
template <matrix A, matrix X, matrix Y>
  requires(
      (__detail::has_csr_base<A> || __detail::has_csc_base<A>) &&
      __detail::has_mdspan_matrix_base<X> && __detail::is_matrix_mdspan_v<Y> &&
      std::is_same_v<typename __detail::ultimate_base_type_t<X>::layout_type,
                     __mdspan::layout_right> &&
      std::is_same_v<typename std::remove_cvref_t<Y>::layout_type,
                     __mdspan::layout_right>)
operation_info_t multiply_inspect(A&& a, X&& x, Y&& y) {
  log_trace("");
  auto info = multiply_inspect(mkl::par, std::forward<A>(a), std::forward<X>(x),
                               std::forward<Y>(y));
  return info;
}

//
// multiply_inspect - CSR/CSC with row major dense matrix rhs without execution
// policy
//
template <matrix A, matrix X, matrix Y>
  requires(
      (__detail::has_csr_base<A> || __detail::has_csc_base<A>) &&
      __detail::has_mdspan_matrix_base<X> && __detail::is_matrix_mdspan_v<Y> &&
      std::is_same_v<typename __detail::ultimate_base_type_t<X>::layout_type,
                     __mdspan::layout_right> &&
      std::is_same_v<typename std::remove_cvref_t<Y>::layout_type,
                     __mdspan::layout_right>)
void multiply_inspect(operation_info_t& info, A&& a, X&& x, Y&& y) {
  log_trace("");
  multiply_inspect(mkl::par, info, std::forward<A>(a), std::forward<X>(x),
                   std::forward<Y>(y));
}

//
// multiply - CSR/CSC with row major dense matrix rhs without execution policy
//
template <matrix A, matrix X, matrix Y>
  requires(
      (__detail::has_csr_base<A> || __detail::has_csc_base<A>) &&
      __detail::has_mdspan_matrix_base<X> && __detail::is_matrix_mdspan_v<Y> &&
      std::is_same_v<typename __detail::ultimate_base_type_t<X>::layout_type,
                     __mdspan::layout_right> &&
      std::is_same_v<typename std::remove_cvref_t<Y>::layout_type,
                     __mdspan::layout_right>)
void multiply(operation_info_t& info, A&& a, X&& x, Y&& y) {
  log_trace("");
  multiply(mkl::par, info, std::forward<A>(a), std::forward<X>(x),
           std::forward<Y>(y));
}

//
// multiply - CSR/CSC with row major dense matrix rhs without execution policy
// or state object
//
template <matrix A, matrix X, matrix Y>
  requires(
      (__detail::has_csr_base<A> || __detail::has_csc_base<A>) &&
      __detail::has_mdspan_matrix_base<X> && __detail::is_matrix_mdspan_v<Y> &&
      std::is_same_v<typename __detail::ultimate_base_type_t<X>::layout_type,
                     __mdspan::layout_right> &&
      std::is_same_v<typename std::remove_cvref_t<Y>::layout_type,
                     __mdspan::layout_right>)
void multiply(A&& a, X&& x, Y&& y) {
  log_trace("");
  operation_info_t info{};
  multiply(mkl::par, info, std::forward<A>(a), std::forward<X>(x),
           std::forward<Y>(y));
}

} // namespace spblas
