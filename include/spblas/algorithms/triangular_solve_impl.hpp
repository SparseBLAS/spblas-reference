#pragma once

#include <complex>
#include <memory_resource>
#include <stdexcept>

#include <spblas/backend/backend.hpp>
#include <spblas/concepts.hpp>
#include <spblas/detail/api_concepts.hpp>
#include <spblas/detail/log.hpp>
#include <spblas/detail/operation_info_t.hpp>
#include <spblas/detail/operation_state.hpp>
#include <spblas/detail/triangular_types.hpp>
#include <spblas/detail/view_analysis.hpp>
#include <spblas/views/triangle_view.hpp>

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

  if constexpr (std::is_same_v<Triangle, upper_triangle_t>) {
    for (auto&& [i, a_row] :
         spblas::__ranges::views::reverse(__backend::rows(a))) {
      V dot_product = 0;
      T diagonal_value = 0;
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
      T diagonal_value = 0;
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

//
// operation_state API (SparseBLAS paper)
//
//   triangular_solve_inspect([policy,] state, T, [alpha,] b, x)
//   triangular_solve([policy,] state, T, [alpha,] b, x)
//     Solves T x = alpha * b for x. x may alias b.
//
//   T: any composition of view::triangle / view::diagonal with
//   view::transpose and view::conjugate that represents a triangular or
//   diagonal matrix, in any order (e.g. view::triangle(A, uplo::lower),
//   view::transpose(view::triangle(A, uplo::lower)),
//   view::triangle(view::triangle(A, uplo::lower), uplo::upper)), also
//   through view::symmetric (a triangle of a symmetric matrix). The stack is
//   reduced by __detail::analyze_view to op(region(A)) at compile time and
//   solved as a triangle of op(A), or as a diagonal solve.
//   view::diag::implicit_zero is rejected at compile time (singular).
//
//   A: CSR (row-oriented substitution) or CSC (column-oriented), and a
//   matrix_opt of either.
//

namespace __detail {

template <typename M, typename B, typename X>
void check_triangular_solve_shapes_(M&& a, B&& b, X&& x) {
  if (__backend::shape(a)[0] != __backend::shape(a)[1] ||
      __backend::shape(a)[0] != __backend::shape(b) ||
      __backend::shape(a)[1] != __backend::shape(x)) {
    throw std::invalid_argument(
        "triangular_solve: matrix and vector dimensions are incompatible.");
  }
}

// Row-oriented substitution with the Upper / lower triangle of a.
template <bool Upper, bool ExplicitDiagonal, typename M, typename Alpha,
          typename B, typename X>
void triangular_solve_rows_(M&& a, Alpha alpha, B&& b, X&& x) {
  using V = tensor_scalar_t<X>;
  const V alpha_v = static_cast<V>(alpha);

  auto solve_row = [&](auto i, auto&& a_row) {
    V dot_product = 0;
    tensor_scalar_t<M> diagonal_value = 0;
    for (auto&& [k, a_v] : a_row) {
      if (Upper ? k > i : k < i) {
        dot_product += a_v * __backend::lookup(x, k);
      } else if (k == i) {
        diagonal_value = a_v;
      }
    }
    V rhs = alpha_v * __backend::lookup(b, i) - dot_product;
    if constexpr (ExplicitDiagonal) {
      __backend::lookup(x, i) = rhs / diagonal_value;
    } else {
      __backend::lookup(x, i) = rhs;
    }
  };

  if constexpr (Upper) {
    for (auto&& [i, a_row] :
         spblas::__ranges::views::reverse(__backend::rows(a))) {
      solve_row(i, a_row);
    }
  } else {
    for (auto&& [i, a_row] : __backend::rows(a)) {
      solve_row(i, a_row);
    }
  }
}

// Column-oriented substitution, for column-iterable matrices (CSC, matrix_opt
// of CSC, the transpose of CSR).
template <bool Upper, bool ExplicitDiagonal, typename M, typename Alpha,
          typename B, typename X>
void triangular_solve_columns_(M&& a, Alpha alpha, B&& b, X&& x) {
  using V = tensor_scalar_t<X>;
  const V alpha_v = static_cast<V>(alpha);

  // x = alpha * b, then eliminate one column at a time.
  for (tensor_index_t<X> i = 0; i < __backend::shape(x); i++) {
    __backend::lookup(x, i) = alpha_v * __backend::lookup(b, i);
  }

  auto solve_column = [&](auto j, auto&& a_column) {
    if constexpr (ExplicitDiagonal) {
      tensor_scalar_t<M> diagonal_value = 0;
      for (auto&& [i, a_v] : a_column) {
        if (i == j) {
          diagonal_value = a_v;
        }
      }
      __backend::lookup(x, j) /= diagonal_value;
    }
    const V x_j = __backend::lookup(x, j);
    for (auto&& [i, a_v] : a_column) {
      if (Upper ? i < j : i > j) {
        __backend::lookup(x, i) -= a_v * x_j;
      }
    }
  };

  if constexpr (Upper) {
    for (auto&& [j, a_column] :
         spblas::__ranges::views::reverse(__backend::columns(a))) {
      solve_column(j, a_column);
    }
  } else {
    for (auto&& [j, a_column] : __backend::columns(a)) {
      solve_column(j, a_column);
    }
  }
}

// x = alpha * inv(D) * b for the represented diagonal D.
template <typename Analysis, typename Alpha, typename B, typename X>
void diagonal_solve_(operation_state& state, const Analysis& a, Alpha alpha,
                     B&& b, X&& x) {
  check_triangular_solve_shapes_(a.base, b, x);
  using V = tensor_scalar_t<X>;
  using T = tensor_scalar_t<typename Analysis::base_type>;
  const V alpha_v = static_cast<V>(alpha);

  std::pmr::vector<T> diagonal(__backend::shape(x), T(0),
                               state.get_allocator().host());
  __detail::for_each_represented_entry(
      a, [&](auto i, auto, auto&& v) { diagonal[i] = v; });
  for (tensor_index_t<X> i = 0; i < __backend::shape(x); i++) {
    __backend::lookup(x, i) = alpha_v * __backend::lookup(b, i) / diagonal[i];
  }
}

template <typename T>
using structured_operand_t =
    decltype(std::declval<view_analysis_t<T>>().operand());

// A triangular structure the reference kernels can iterate: its operand
// op(base) is row- or column-iterable (diagonal regions need no operand).
template <typename T>
constexpr bool is_solvable_structure_() {
  if constexpr (triangular_structure<T>) {
    using A = view_analysis_t<T>;
    using Op = structured_operand_t<T>;
    return A::region == view_region::diagonal || __backend::row_iterable<Op> ||
           __backend::column_iterable<Op>;
  } else {
    return false;
  }
}

template <typename T>
concept solvable_structure = is_solvable_structure_<T>();

} // namespace __detail

template <typename T, __detail::scalar Alpha, vector B, vector X>
  requires(__detail::solvable_structure<T> && __backend::lookupable<B> &&
           __backend::lookupable<X>)
void triangular_solve(operation_state& state, T&& t, Alpha alpha, B&& b,
                      X&& x) {
  log_trace("");
  auto a = __detail::analyze_view(t);
  using A = decltype(a);

  if constexpr (A::region == __detail::view_region::diagonal) {
    __detail::diagonal_solve_(state, a, alpha, b, x);
  } else {
    constexpr bool upper =
        A::represented_region == __detail::view_region::upper;
    constexpr bool explicit_diagonal =
        A::diag == __detail::diag_kind::explicit_diagonal;
    auto op = a.operand();
    __detail::check_triangular_solve_shapes_(op, b, x);
    if constexpr (__backend::row_iterable<decltype(op)>) {
      __detail::triangular_solve_rows_<upper, explicit_diagonal>(op, alpha, b,
                                                                 x);
    } else {
      __detail::triangular_solve_columns_<upper, explicit_diagonal>(op, alpha,
                                                                    b, x);
    }
  }
}

// An implied zero diagonal makes the matrix singular.
template <typename T, __detail::scalar Alpha, vector B, vector X>
  requires(__detail::is_implicit_zero_structure_<T>())
void triangular_solve(operation_state& state, T&& t, Alpha alpha, B&& b,
                      X&& x) {
  static_assert(!__detail::is_implicit_zero_structure_<T>(),
                "triangular_solve: view::diag::implicit_zero is not supported "
                "(the matrix would be singular).");
}

template <typename T, vector B, vector X>
  requires(__detail::solvable_structure<T> ||
           __detail::is_implicit_zero_structure_<T>())
void triangular_solve(operation_state& state, T&& t, B&& b, X&& x) {
  log_trace("");
  triangular_solve(state, std::forward<T>(t), tensor_scalar_t<X>(1),
                   std::forward<B>(b), std::forward<X>(x));
}

// The reference implementation has nothing to prepare.
template <typename T, __detail::scalar Alpha, vector B, vector X>
  requires(__detail::solvable_structure<T>)
void triangular_solve_inspect(operation_state& state, T&& t, Alpha alpha, B&& b,
                              X&& x) {
  log_trace("");
}

template <typename T, vector B, vector X>
  requires(__detail::solvable_structure<T>)
void triangular_solve_inspect(operation_state& state, T&& t, B&& b, X&& x) {
  log_trace("");
}

} // namespace spblas

// The policy-first overloads (reference: the policy is ignored).
#include <spblas/algorithms/policy_forwarding.hpp>
