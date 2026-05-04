#pragma once

#include <oneapi/mkl.hpp>

#include <spblas/detail/log.hpp>
#include <spblas/detail/operation_info_t.hpp>
#include <spblas/detail/ranges.hpp>
#include <spblas/detail/view_inspectors.hpp>

#include <spblas/detail/triangular_types.hpp>
#include <spblas/views/matrix_view.hpp>

namespace spblas {

//  Mappings from Triangular Solve input args to oneMKL vendor input args
//
//  using   A = L + D + U as a strict decomposition of triangular parts
//
//  spblas_ref input            ->   oneMKL SpTRSV input
//  uplo(op(A))                 ->   op(uplo(A))
//
//  upper + nontrans  (D+U)     ->   nontrans  + upper (D+U)
//  lower + nontrans  (L+D)     ->   nontrans  + lower (L+D)
//  upper + trans     (L+D)^T   ->   trans     + lower (L+D)^T
//  lower + trans     (D+U)^T   ->   trans     + upper (D+U)^T
//  upper + conjtrans (L+D)^H   ->   conjtrans + lower (L+D)^H
//  lower + conjtrans (D+U)^H   ->   conjtrans + upper (D+U)^H
//

//
// CSR triangular solve inspection step
//
template <typename ExecutionPolicy, matrix A, class Triangle,
          class DiagonalStorage, vector B, vector X>
  requires __detail::has_csr_base<A> &&
           __detail::has_contiguous_range_base<B> &&
           __ranges::contiguous_range<X>
void triangular_solve_inspect(ExecutionPolicy&& policy, A&& a, Triangle uplo,
                              DiagonalStorage diag, B&& b, X&& x) {
  log_trace("");
  static_assert(std::is_same_v<Triangle, upper_triangle_t> ||
                std::is_same_v<Triangle, lower_triangle_t>);
  static_assert(std::is_same_v<DiagonalStorage, explicit_diagonal_t> ||
                std::is_same_v<DiagonalStorage, implicit_unit_diagonal_t>);

  if (__detail::is_conjugated(b) || __detail::is_conjugated(x)) {
    throw std::runtime_error(
        "oneMKL SYCL backend does not support conjugated dense vectors.");
  }

  using T = tensor_scalar_t<A>;
  using I = tensor_index_t<A>;
  using O = tensor_offset_t<A>;

  auto a_data = __detail::get_ultimate_base(a).values().data();
  auto&& q = __mkl::get_queue(policy, a_data);

  auto a_handle = __mkl::get_matrix_handle(q, a);
  auto a_op = __mkl::get_transpose(a);

  auto uplo_val =
      std::is_same_v<Triangle, upper_triangle_t>
          ? oneapi::mkl::uplo::upper
          : oneapi::mkl::uplo::lower; // someday apply mapping with op

  auto diag_val = std::is_same_v<DiagonalStorage, explicit_diagonal_t>
                      ? oneapi::mkl::diag::nonunit
                      : oneapi::mkl::diag::unit;

  oneapi::mkl::sparse::optimize_trsv(q, uplo_val, a_op, diag_val, a_handle)
      .wait();

  if (!__detail::has_matrix_opt(a)) {
    oneapi::mkl::sparse::release_matrix_handle(q, &a_handle).wait();
  }
}

template <typename ExecutionPolicy, matrix A, vector B, vector X>
  requires __detail::has_csr_base<A> &&
           __detail::has_contiguous_range_base<B> &&
           __ranges::contiguous_range<X>
void triangular_solve_inspect(ExecutionPolicy&& policy, A&& a, B&& b, X&& x) {
  log_trace("");
  using type = decltype(matrix_view::legacy_pattern(a));
  triangular_solve_inspect(
      policy, __detail::get_ultimate_base_or_matrix_opt(a),
      std::conditional_t<
          std::is_same_v<typename type::uplo, matrix_view::uplo::upper>,
          upper_triangle_t, lower_triangle_t>{},
      std::conditional_t<
          std::is_same_v<typename type::diag, matrix_view::diag::explicit_diag>,
          explicit_diagonal_t, implicit_unit_diagonal_t>{},
      b, x);
}

//
// CSR triangular solve execution step
//
template <typename ExecutionPolicy, matrix A, class Triangle,
          class DiagonalStorage, vector B, vector X>
  requires __detail::has_csr_base<A> &&
           __detail::has_contiguous_range_base<B> &&
           __ranges::contiguous_range<X>
void triangular_solve(ExecutionPolicy&& policy, A&& a, Triangle uplo,
                      DiagonalStorage diag, B&& b, X&& x) {
  log_trace("");
  static_assert(std::is_same_v<Triangle, upper_triangle_t> ||
                std::is_same_v<Triangle, lower_triangle_t>);
  static_assert(std::is_same_v<DiagonalStorage, explicit_diagonal_t> ||
                std::is_same_v<DiagonalStorage, implicit_unit_diagonal_t>);

  if (__detail::is_conjugated(b) || __detail::is_conjugated(x)) {
    throw std::runtime_error(
        "oneMKL SYCL backend does not support conjugated dense vectors.");
  }

  using T = tensor_scalar_t<A>;
  using I = tensor_index_t<A>;
  using O = tensor_offset_t<A>;

  auto alpha_optional = __detail::get_scaling_factor(a, b);
  tensor_scalar_t<A> alpha = alpha_optional.value_or(1);

  auto a_data = __detail::get_ultimate_base(a).values().data();
  auto&& q = __mkl::get_queue(policy, a_data);

  auto a_handle = __mkl::get_matrix_handle(q, a);
  auto a_op = __mkl::get_transpose(a);

  auto uplo_val = std::is_same_v<Triangle, upper_triangle_t>
                      ? oneapi::mkl::uplo::upper
                      : oneapi::mkl::uplo::lower;

  auto diag_val = std::is_same_v<DiagonalStorage, explicit_diagonal_t>
                      ? oneapi::mkl::diag::nonunit
                      : oneapi::mkl::diag::unit;

  auto b_base = __detail::get_ultimate_base(b);

  oneapi::mkl::sparse::trsv(q, uplo_val, a_op, diag_val, alpha, a_handle,
                            __ranges::data(b_base), __ranges::data(x))
      .wait();

  if (!__detail::has_matrix_opt(a)) {
    oneapi::mkl::sparse::release_matrix_handle(q, &a_handle).wait();
  }

} // triangular_solve

template <typename ExecutionPolicy, matrix A, vector B, vector X>
  requires __detail::has_csr_base<A> &&
           __detail::has_contiguous_range_base<B> &&
           __ranges::contiguous_range<X> && __detail::has_legacy_pattern_d<A> &&
           (!std::is_same_v<
               typename __detail::ultimate_base_or_matrix_type_t<A>::diag,
               matrix_view::diag::implicit_zero>) &&
           (!std::is_same_v<
               typename __detail::ultimate_base_or_matrix_type_t<A>::uplo,
               matrix_view::uplo::full>)
void triangular_solve(ExecutionPolicy&& policy, A&& a, B&& b, X&& x) {
  log_trace("");
  using type = decltype(matrix_view::legacy_pattern(a));
  triangular_solve(
      policy, __detail::get_ultimate_base_or_matrix_opt(a),
      std::conditional_t<
          std::is_same_v<typename type::uplo, matrix_view::uplo::upper>,
          upper_triangle_t, lower_triangle_t>{},
      std::conditional_t<
          std::is_same_v<typename type::diag, matrix_view::diag::explicit_diag>,
          explicit_diagonal_t, implicit_unit_diagonal_t>{},
      b, x);
}

//
// CSR triangular_solve_inspect with no exception policy
//
template <matrix A, class Triangle, class DiagonalStorage, vector B, vector X>
  requires __detail::has_csr_base<A> &&
           __detail::has_contiguous_range_base<B> &&
           __ranges::contiguous_range<X>
void triangular_solve_inspect(A&& a, Triangle uplo, DiagonalStorage diag, B&& b,
                              X&& x) {
  triangular_solve_inspect(mkl::par, std::forward<A>(a),
                           std::forward<Triangle>(uplo),
                           std::forward<DiagonalStorage>(diag),
                           std::forward<B>(b), std::forward<X>(x));
} // triangular_solve_inspect

//
// CSR triangular_solve with no exception policy
//
template <matrix A, class Triangle, class DiagonalStorage, vector B, vector X>
  requires __detail::has_csr_base<A> &&
           __detail::has_contiguous_range_base<B> &&
           __ranges::contiguous_range<X>
void triangular_solve(A&& a, Triangle uplo, DiagonalStorage diag, B&& b,
                      X&& x) {
  triangular_solve(mkl::par, std::forward<A>(a), std::forward<Triangle>(uplo),
                   std::forward<DiagonalStorage>(diag), std::forward<B>(b),
                   std::forward<X>(x));
} // triangular_solve

template <matrix A, vector B, vector X>
  requires __detail::has_csr_base<A> &&
           __detail::has_contiguous_range_base<B> &&
           __ranges::contiguous_range<X> && __detail::has_legacy_pattern_d<A> &&
           (!std::is_same_v<
               typename __detail::ultimate_base_or_matrix_type_t<A>::diag,
               matrix_view::diag::implicit_zero>) &&
           (!std::is_same_v<
               typename __detail::ultimate_base_or_matrix_type_t<A>::uplo,
               matrix_view::uplo::full>)
void triangular_solve(A&& a, B&& b, X&& x) {
  triangular_solve(mkl::par, std::forward<A>(a), std::forward<B>(b),
                   std::forward<X>(x));
}

} // namespace spblas
