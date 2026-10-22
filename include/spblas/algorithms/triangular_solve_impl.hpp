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
    for (auto&& [i, a_row] : __ranges::views::reverse(__backend::rows(a))) {
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
//   triangular_solve_inspect([policy,] state, view::triangle(A, uplo, diag),
//                            [alpha,] b, x)
//   triangular_solve([policy,] state, view::triangle(A, uplo, diag),
//                    [alpha,] b, x)
//     Solves T x = alpha * b for x, where T is the selected triangle of A.
//     x may alias b. view::diag::implicit_zero is rejected (T is singular).
//
//   A: CSR (row-oriented substitution) or CSC (column-oriented), and a
//   matrix_opt of either.
//
//   Any composition of view::triangle / view::diagonal with view::transpose
//   and view::conjugate that represents a triangular or diagonal matrix is
//   accepted, in any order (e.g. view::transpose(view::triangle(A, lower)),
//   view::triangle(view::triangle(A, lower), upper)): the stack is reduced
//   by __detail::analyze_view to op(region(A)) and solved as
//   view::triangle(op(A), ...) or as a diagonal solve.
//

namespace __detail {

template <typename T>
using triangle_matrix_t = decltype(std::declval<T>().matrix());

template <typename T, typename M, typename B, typename X>
void check_triangular_solve_(T&& t, M&& a, B&& b, X&& x) {
  if (t.diag() == view::diag::implicit_zero) {
    throw std::invalid_argument(
        "triangular_solve: view::diag::implicit_zero is not supported (the "
        "triangle would be singular).");
  }
  if (__backend::shape(a)[0] != __backend::shape(a)[1] ||
      __backend::shape(a)[0] != __backend::shape(b) ||
      __backend::shape(a)[1] != __backend::shape(x)) {
    throw std::invalid_argument(
        "triangular_solve: matrix and vector dimensions are incompatible.");
  }
}

} // namespace __detail

template <typename T, __detail::scalar Alpha, vector B, vector X>
  requires(__detail::is_triangle_view_v<T> &&
           __backend::row_iterable<__detail::triangle_matrix_t<T>> &&
           __backend::lookupable<B> && __backend::lookupable<X>)
void triangular_solve(operation_state& state, T&& t, Alpha alpha, B&& b,
                      X&& x) {
  log_trace("");
  auto a = t.matrix();
  __detail::check_triangular_solve_(t, a, b, x);

  using V = tensor_scalar_t<X>;
  const V alpha_v = static_cast<V>(alpha);
  const bool upper = t.uplo() == view::uplo::upper;
  const bool explicit_diagonal = t.diag() == view::diag::explicit_diagonal;

  auto solve_row = [&](auto i, auto&& a_row) {
    V dot_product = 0;
    tensor_scalar_t<__detail::triangle_matrix_t<T>> diagonal_value = 0;
    for (auto&& [k, a_v] : a_row) {
      if (upper ? k > i : k < i) {
        dot_product += a_v * __backend::lookup(x, k);
      } else if (k == i) {
        diagonal_value = a_v;
      }
    }
    V rhs = alpha_v * __backend::lookup(b, i) - dot_product;
    __backend::lookup(x, i) = explicit_diagonal ? rhs / diagonal_value : rhs;
  };

  if (upper) {
    for (auto&& [i, a_row] : __ranges::views::reverse(__backend::rows(a))) {
      solve_row(i, a_row);
    }
  } else {
    for (auto&& [i, a_row] : __backend::rows(a)) {
      solve_row(i, a_row);
    }
  }
}

// Column-oriented substitution, for triangles of column-iterable matrices
// (CSC, matrix_opt of CSC).
template <typename T, __detail::scalar Alpha, vector B, vector X>
  requires(__detail::is_triangle_view_v<T> &&
           !__backend::row_iterable<__detail::triangle_matrix_t<T>> &&
           __backend::column_iterable<__detail::triangle_matrix_t<T>> &&
           __backend::lookupable<B> && __backend::lookupable<X>)
void triangular_solve(operation_state& state, T&& t, Alpha alpha, B&& b,
                      X&& x) {
  log_trace("");
  auto a = t.matrix();
  __detail::check_triangular_solve_(t, a, b, x);

  using V = tensor_scalar_t<X>;
  const V alpha_v = static_cast<V>(alpha);
  const bool upper = t.uplo() == view::uplo::upper;
  const bool explicit_diagonal = t.diag() == view::diag::explicit_diagonal;

  // x = alpha * b, then eliminate one column at a time.
  for (tensor_index_t<X> i = 0; i < __backend::shape(x); i++) {
    __backend::lookup(x, i) = alpha_v * __backend::lookup(b, i);
  }

  auto solve_column = [&](auto j, auto&& a_column) {
    if (explicit_diagonal) {
      tensor_scalar_t<__detail::triangle_matrix_t<T>> diagonal_value = 0;
      for (auto&& [i, a_v] : a_column) {
        if (i == j) {
          diagonal_value = a_v;
        }
      }
      __backend::lookup(x, j) /= diagonal_value;
    }
    const V x_j = __backend::lookup(x, j);
    for (auto&& [i, a_v] : a_column) {
      if (upper ? i < j : i > j) {
        __backend::lookup(x, i) -= a_v * x_j;
      }
    }
  };

  if (upper) {
    for (auto&& [j, a_column] :
         __ranges::views::reverse(__backend::columns(a))) {
      solve_column(j, a_column);
    }
  } else {
    for (auto&& [j, a_column] : __backend::columns(a)) {
      solve_column(j, a_column);
    }
  }
}

namespace __detail {

// A stack with structure that is not simply view::triangle over an
// unstructured operand (which the kernels above take directly).
template <typename T>
constexpr bool is_plain_triangle_() {
  if constexpr (is_triangle_view_v<T>) {
    return !structured_view<triangle_matrix_t<T>>;
  } else {
    return false;
  }
}

template <typename T>
concept composed_structure = structured_view<T> && !is_plain_triangle_<T>();

template <typename T>
using structured_operand_t =
    decltype(std::declval<view_analysis_t<T>>().operand());

// x = alpha * inv(D) * b for the diagonal D of op(base).
template <typename Analysis, typename Alpha, typename B, typename X>
void diagonal_solve_(operation_state& state, const Analysis& a, Alpha alpha,
                     B&& b, X&& x) {
  const auto shape = __backend::shape(a.base);
  if (a.diag == spblas::view::diag::implicit_zero) {
    throw std::invalid_argument(
        "triangular_solve: view::diag::implicit_zero is not supported (the "
        "matrix would be singular).");
  }
  if (shape[0] != shape[1] || shape[0] != __backend::shape(b) ||
      shape[1] != __backend::shape(x)) {
    throw std::invalid_argument(
        "triangular_solve: matrix and vector dimensions are incompatible.");
  }

  using V = tensor_scalar_t<X>;
  const V alpha_v = static_cast<V>(alpha);
  if (a.diag == spblas::view::diag::implicit_unit) {
    for (tensor_index_t<X> i = 0; i < __backend::shape(x); i++) {
      __backend::lookup(x, i) = alpha_v * __backend::lookup(b, i);
    }
    return;
  }

  // The diagonal of op(base) is that of base, conjugated if op conjugates.
  using T = tensor_scalar_t<typename Analysis::base_type>;
  std::pmr::vector<T> diagonal(shape[0], T(0), state.get_allocator());
  __backend::for_each(a.base, [&](auto&& e) {
    auto&& [idx, v] = e;
    auto&& [i, j] = idx;
    if (i == j) {
      if constexpr (Analysis::conjugate && is_std_complex_v<T>) {
        diagonal[i] = std::conj(v);
      } else {
        diagonal[i] = v;
      }
    }
  });
  for (tensor_index_t<X> i = 0; i < __backend::shape(x); i++) {
    __backend::lookup(x, i) = alpha_v * __backend::lookup(b, i) / diagonal[i];
  }
}

} // namespace __detail

// Composed views: solved through their analysis (see the top of the file).
template <typename T, __detail::scalar Alpha, vector B, vector X>
  requires(__detail::composed_structure<T> &&
           (__backend::row_iterable<__detail::structured_operand_t<T>> ||
            __backend::column_iterable<__detail::structured_operand_t<T>>) &&
           __backend::lookupable<B> && __backend::lookupable<X>)
void triangular_solve(operation_state& state, T&& t, Alpha alpha, B&& b,
                      X&& x) {
  log_trace("");
  auto a = __detail::analyze_view(t);
  switch (a.represented_region()) {
  case __detail::view_region::lower:
  case __detail::view_region::upper: {
    const auto uplo = a.represented_region() == __detail::view_region::lower
                          ? view::uplo::lower
                          : view::uplo::upper;
    triangular_solve(state, view::triangle(a.operand(), uplo, a.diag), alpha,
                     std::forward<B>(b), std::forward<X>(x));
    break;
  }
  case __detail::view_region::diagonal:
    __detail::diagonal_solve_(state, a, alpha, b, x);
    break;
  case __detail::view_region::full:
    // A structured stack always restricts the region.
    throw std::logic_error("triangular_solve: unexpected full region.");
  }
}

template <typename T, vector B, vector X>
  requires(__detail::structured_view<T>)
void triangular_solve(operation_state& state, T&& t, B&& b, X&& x) {
  log_trace("");
  triangular_solve(state, std::forward<T>(t), tensor_scalar_t<X>(1),
                   std::forward<B>(b), std::forward<X>(x));
}

// The reference implementation has nothing to prepare.
template <typename T, __detail::scalar Alpha, vector B, vector X>
  requires(__detail::structured_view<T>)
void triangular_solve_inspect(operation_state& state, T&& t, Alpha alpha, B&& b,
                              X&& x) {
  log_trace("");
}

template <typename T, vector B, vector X>
  requires(__detail::structured_view<T>)
void triangular_solve_inspect(operation_state& state, T&& t, B&& b, X&& x) {
  log_trace("");
}

} // namespace spblas

// The policy-first overloads (reference: the policy is ignored).
#include <spblas/algorithms/policy_forwarding.hpp>
