#pragma once

#include <spblas/backend/backend.hpp>
#include <spblas/concepts.hpp>
#include <spblas/detail/log.hpp>
#include <spblas/detail/operation_info_t.hpp>
#include <spblas/detail/triangular_types.hpp>

namespace spblas {

template <matrix A, class Triangle, class DiagonalStorage, vector B, vector X>
  requires(__backend::row_iterable<A> && __backend::lookupable<B> &&
           __backend::lookupable<X>)
void triangular_solve(A&& a, Triangle t, DiagonalStorage d, B&& b, X&& x) {
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

} // namespace spblas
