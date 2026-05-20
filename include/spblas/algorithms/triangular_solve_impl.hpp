#pragma once

#include <spblas/backend/backend.hpp>
#include <spblas/concepts.hpp>
#include <spblas/detail/log.hpp>
#include <spblas/detail/operation_info_t.hpp>
#include <spblas/detail/triangular_types.hpp>
#include <spblas/views/matrix_view.hpp>

namespace spblas {

// X = inv(A) B
// SpTRSV inspect stage
template <matrix A, class Triangle, class DiagonalStorage, vector B, vector X>
  requires(__backend::row_iterable<A> && __backend::lookupable<B> &&
           __backend::lookupable<X>)
operation_info_t triangular_solve_inspect(A&& a, Triangle t, DiagonalStorage d,
                                          B&& b, X&& x) {
  log_trace("");
  static_assert(std::is_same_v<Triangle, upper_triangle_t> ||
                std::is_same_v<Triangle, lower_triangle_t>);
  assert(__backend::shape(a)[0] == __backend::shape(a)[1]);

  return operation_info_t{};
}

template <matrix A, vector B, vector X>
operation_info_t triangular_solve_inspect(ExecutionPolicy&& policy, A&& a,
                                          B&& b, X&& x) {
  log_trace("");
  using type = decltype(matrix_view::legacy_pattern(a));
  triangular_solve_inspect(
      __detail::get_ultimate_base_or_matrix_opt(a),
      std::conditional_t<
          std::is_same_v<typename type::uplo, matrix_view::uplo::upper>,
          upper_triangle_t, lower_triangle_t>{},
      std::conditional_t<
          std::is_same_v<typename type::diag, matrix_view::diag::explicit_diag>,
          explicit_diagonal_t, implicit_unit_diagonal_t>{},
      b, x);
}

// X = inv(A) B
// SpTRSV inspect stage
template <matrix A, class Triangle, class DiagonalStorage, vector B, vector X>
  requires(__backend::row_iterable<A> && __backend::lookupable<B> &&
           __backend::lookupable<X>)
void triangular_solve_inspect(operation_info_t& info, A&& a, Triangle t,
                              DiagonalStorage d, B&& b, X&& x) {
  log_trace("");
  static_assert(std::is_same_v<Triangle, upper_triangle_t> ||
                std::is_same_v<Triangle, lower_triangle_t>);
  assert(__backend::shape(a)[0] == __backend::shape(a)[1]);
}

template <matrix A, vector B, vector X>
void triangular_solve_inspect(operation_info_t& info, A&& a, B&& b, X&& x) {
  log_trace("");
  using type = decltype(matrix_view::legacy_pattern(a));
  triangular_solve_inspect(
      info, __detail::get_ultimate_base_or_matrix_opt(a),
      std::conditional_t<
          std::is_same_v<typename type::uplo, matrix_view::uplo::upper>,
          upper_triangle_t, lower_triangle_t>{},
      std::conditional_t<
          std::is_same_v<typename type::diag, matrix_view::diag::explicit_diag>,
          explicit_diagonal_t, implicit_unit_diagonal_t>{},
      b, x);
}

// X = inv(A) B
// SpTRSV solve stage
template <matrix A, class Triangle, class DiagonalStorage, vector B, vector X>
  requires(__backend::row_iterable<A> && __backend::lookupable<B> &&
           __backend::lookupable<X>)
void triangular_solve(A&& a, Triangle t, DiagonalStorage d, B&& b, X&& x) {
  log_trace("");
  static_assert(std::is_same_v<Triangle, upper_triangle_t> ||
                std::is_same_v<Triangle, lower_triangle_t>);
  assert(__backend::shape(a)[0] == __backend::shape(a)[1]);

  assert(__backend::shape(a)[1] == __backend::shape(x) &&
         __backend::shape(a)[0] == __backend::shape(b));

  using T = tensor_scalar_t<A>;
  using V = decltype(std::declval<tensor_scalar_t<A>>() *
                     std::declval<tensor_scalar_t<X>>());

  T diagonal_value = 0;

  if constexpr (std::is_same_v<Triangle, upper_triangle_t>) {
    for (auto&& [i, a_row] : __ranges::views::reverse(__backend::rows(a))) {
      V dot_product = 0;
      for (auto&& [k, a_v] : a_row) {
        if (k > i) {
          dot_product += a_v * __backend::lookup(x, k);
        } else if (i == k) {
          diagonal_value = a_v;
        }
      }
      if constexpr (std::is_same_v<DiagonalStorage, explicit_diagonal_t>) {
        __backend::lookup(x, i) =
            (__backend::lookup(b, i) - dot_product) / diagonal_value;
      } else {
        __backend::lookup(x, i) = __backend::lookup(b, i) - dot_product;
      }
    }
  } else if constexpr (std::is_same_v<Triangle, lower_triangle_t>) {
    for (auto&& [i, a_row] : __backend::rows(a)) {
      V dot_product = 0;
      for (auto&& [k, a_v] : a_row) {
        if (k < i) {
          dot_product += a_v * __backend::lookup(x, k);
        } else if (i == k) {
          diagonal_value = a_v;
        }
      }
      if constexpr (std::is_same_v<DiagonalStorage, explicit_diagonal_t>) {
        __backend::lookup(x, i) =
            (__backend::lookup(b, i) - dot_product) / diagonal_value;
      } else {
        __backend::lookup(x, i) = __backend::lookup(b, i) - dot_product;
      }
    }
  }
}

// X = inv(A) B
// SpTRSV solve stage with info
template <matrix A, vector B, vector X>
  requires __detail::has_legacy_pattern_d<A> &&
           (!std::is_same_v<
               typename __detail::ultimate_base_or_matrix_type_t<A>::diag,
               matrix_view::diag::implicit_zero>) &&
           (!std::is_same_v<
               typename __detail::ultimate_base_or_matrix_type_t<A>::uplo,
               matrix_view::uplo::full>)
void triangular_solve(A&& a, B&& b, X&& x) {
  log_trace("");
  using type = decltype(matrix_view::legacy_pattern(a));
  triangular_solve(
      __detail::get_ultimate_base_or_matrix_opt(a),
      std::conditional_t<
          std::is_same_v<typename type::uplo, matrix_view::uplo::upper>,
          upper_triangle_t, lower_triangle_t>{},
      std::conditional_t<
          std::is_same_v<typename type::diag, matrix_view::diag::explicit_diag>,
          explicit_diagonal_t, implicit_unit_diagonal_t>{},
      b, x);
}

template <matrix A, class Triangle, class DiagonalStorage, vector B, vector X>
  requires(__backend::row_iterable<A> && __backend::lookupable<B> &&
           __backend::lookupable<X>)
void triangular_solve(operation_info_t& info, A&& a, Triangle t,
                      DiagonalStorage d, B&& b, X&& x) {
  log_trace("");
  triangular_solve(std::forward<A>(a), std::forward<Triangle>(t),
                   std::forward<DiagonalStorage>(d), std::forward<B>(b),
                   std::forward<X>(x));
}

} // namespace spblas
