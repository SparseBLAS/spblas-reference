#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <type_traits>
#include <utility>

#include "util.hpp"
#include <spblas/spblas.hpp>

namespace {

// Random n x n CSR matrix with a dominant stored diagonal, so every triangle
// is well conditioned.
template <typename T, typename I>
struct triangular_test_matrix {
  std::vector<T> values;
  std::vector<I> rowptr;
  std::vector<I> colind;
  std::size_t n;

  triangular_test_matrix(std::size_t n_, std::size_t nnz) : n(n_) {
    auto [v, r, c, s, z] = spblas::generate_csr<T, I>(n, n, nnz);
    rowptr.push_back(0);
    for (std::size_t i = 0; i < n; i++) {
      for (auto p = r[i]; p < r[i + 1]; p++) {
        if (c[p] != i) {
          colind.push_back(c[p]);
          values.push_back(v[p] / T(100 * n));
        }
      }
      colind.push_back(i);
      values.push_back(T(2) + T(i % 3));
      rowptr.push_back(colind.size());
    }
  }

  spblas::csr_wrapper<T, I> csr() {
    return spblas::csr_wrapper<T, I>(values, rowptr, colind, {I(n), I(n)},
                                     I(values.size()));
  }

  // The same matrix in CSC: the CSR arrays of its transpose.
  std::vector<T> csc_values;
  std::vector<I> colptr;
  std::vector<I> rowind;

  spblas::csc_wrapper<T, I> csc() {
    if (colptr.empty()) {
      csc_values.resize(values.size());
      rowind.resize(values.size());
      colptr.resize(n + 1);
      spblas::csr_wrapper<T, I> a_t(csc_values, colptr, rowind, {I(n), I(n)},
                                    I(values.size()));
      spblas::operation_state state;
      spblas::transpose(state, csr(), a_t);
    }
    return spblas::csc_wrapper<T, I>(csc_values, colptr, rowind, {I(n), I(n)},
                                     I(values.size()));
  }

  // r = T * x, with T the selected triangle of the matrix.
  template <typename Uplo, typename Diag>
  std::vector<T> apply(const std::vector<T>& x, Uplo, Diag) const {
    constexpr bool lower = std::is_same_v<Uplo, spblas::view::uplo::lower_t>;
    constexpr bool explicit_diagonal =
        std::is_same_v<Diag, spblas::view::diag::explicit_diagonal_t>;
    constexpr bool implicit_unit =
        std::is_same_v<Diag, spblas::view::diag::implicit_unit_t>;
    std::vector<T> r(n, T(0));
    for (std::size_t i = 0; i < n; i++) {
      for (auto p = rowptr[i]; p < rowptr[i + 1]; p++) {
        std::size_t j = colind[p];
        bool in_triangle = lower ? j < i : j > i;
        if (in_triangle || (j == i && explicit_diagonal)) {
          r[i] += values[p] * x[j];
        }
      }
      if (implicit_unit) {
        r[i] += x[i];
      }
    }
    return r;
  }
};

template <typename T>
void expect_close(const std::vector<T>& expected,
                  const std::vector<T>& actual) {
  ASSERT_EQ(expected.size(), actual.size());
  for (std::size_t i = 0; i < expected.size(); i++) {
    EXPECT_NEAR(expected[i], actual[i], 1e-10 * (1 + std::abs(expected[i])));
  }
}

template <bool Csc, typename Uplo, typename Diag>
void triangular_solve_test(Uplo uplo, Diag diag) {
  using T = double;
  using I = spblas::index_t;

  for (auto&& [m, n, nnz] : util::square_dims) {
    for (T alpha : {T(1), T(-0.5)}) {
      triangular_test_matrix<T, I> a(n, nnz);
      auto b = util::make_values<T>(n);
      std::vector<T> x(n, T(0));

      spblas::operation_state state;
      auto t = [&] {
        if constexpr (Csc) {
          return spblas::view::triangle(a.csc(), uplo, diag);
        } else {
          return spblas::view::triangle(a.csr(), uplo, diag);
        }
      }();
      spblas::triangular_solve_inspect(state, t, alpha, b, x);
      spblas::triangular_solve(state, t, alpha, b, x);

      // Residual check: T * x == alpha * b.
      std::vector<T> alpha_b(n);
      for (std::size_t i = 0; i < n; i++) {
        alpha_b[i] = alpha * b[i];
      }
      expect_close(alpha_b, a.apply(x, uplo, diag));
    }
  }
}

} // namespace

TEST(TriangularSolve, LowerExplicit) {
  triangular_solve_test<false>(spblas::view::uplo::lower,
                               spblas::view::diag::explicit_diagonal);
}

TEST(TriangularSolve, CscLowerExplicit) {
  triangular_solve_test<true>(spblas::view::uplo::lower,
                              spblas::view::diag::explicit_diagonal);
}

TEST(TriangularSolve, UpperExplicit) {
  triangular_solve_test<false>(spblas::view::uplo::upper,
                               spblas::view::diag::explicit_diagonal);
}

TEST(TriangularSolve, CscUpperExplicit) {
  triangular_solve_test<true>(spblas::view::uplo::upper,
                              spblas::view::diag::explicit_diagonal);
}

TEST(TriangularSolve, LowerImplicitUnit) {
  triangular_solve_test<false>(spblas::view::uplo::lower,
                               spblas::view::diag::implicit_unit);
}

TEST(TriangularSolve, CscLowerImplicitUnit) {
  triangular_solve_test<true>(spblas::view::uplo::lower,
                              spblas::view::diag::implicit_unit);
}

TEST(TriangularSolve, UpperImplicitUnit) {
  triangular_solve_test<false>(spblas::view::uplo::upper,
                               spblas::view::diag::implicit_unit);
}

TEST(TriangularSolve, CscUpperImplicitUnit) {
  triangular_solve_test<true>(spblas::view::uplo::upper,
                              spblas::view::diag::implicit_unit);
}

TEST(TriangularSolve, DefaultAlphaMatrixOptPolicyInPlace) {
  using T = double;
  using I = spblas::index_t;

  auto [m, n, nnz] = util::square_dims[1];
  triangular_test_matrix<T, I> a(n, nnz);
  spblas::matrix_opt a_opt(a.csr());
  auto b = util::make_values<T>(n);

  // Without alpha (alpha = 1), through a matrix_opt and a policy, solving in
  // place (x aliases b).
  std::vector<T> x = b;
  spblas::operation_state state;
  auto t = spblas::view::triangle(a_opt, spblas::view::uplo::lower);
  spblas::triangular_solve_inspect(util::test_policy{}, state, t, x, x);
  spblas::triangular_solve(util::test_policy{}, state, t, x, x);

  expect_close(b, a.apply(x, spblas::view::uplo::lower,
                          spblas::view::diag::explicit_diagonal));
}

// An implied zero diagonal (a singular triangle) is rejected at compile time;
// triangular_solve with it is a static_assert.
TEST(TriangularSolve, ImplicitZeroRejected) {
  using T = double;
  using I = spblas::index_t;
  using A = spblas::csr_wrapper<T, I>;

  using Zero = decltype(spblas::view::triangle(
      std::declval<A>(), spblas::view::uplo::lower,
      spblas::view::diag::implicit_zero));
  using Unit = decltype(spblas::view::triangle(
      std::declval<A>(), spblas::view::uplo::lower,
      spblas::view::diag::implicit_unit));
  static_assert(!spblas::__detail::solvable_structure<Zero>);
  static_assert(spblas::__detail::solvable_structure<Unit>);
}

TEST(TriangularSolve, DimensionMismatchThrows) {
  using T = double;
  using I = spblas::index_t;

  auto [values, rowptr, colind, shape, nnz] =
      spblas::generate_csr<T, I>(10, 12, 20);
  spblas::csr_wrapper<T, I> rect(values, rowptr, colind, shape, nnz);
  triangular_test_matrix<T, I> a(10, 20);
  std::vector<T> b(10), x(10), x_bad(9);
  spblas::operation_state state;

  EXPECT_THROW(
      spblas::triangular_solve(
          state, spblas::view::triangle(rect, spblas::view::uplo::lower), b, x),
      std::invalid_argument);
  EXPECT_THROW(spblas::triangular_solve(
                   state,
                   spblas::view::triangle(a.csr(), spblas::view::uplo::upper),
                   b, x_bad),
               std::invalid_argument);
}

TEST(TriangularSolve, CscMatrixOptPolicyInPlace) {
  using T = double;
  using I = spblas::index_t;

  auto [m, n, nnz] = util::square_dims[1];
  triangular_test_matrix<T, I> a(n, nnz);
  spblas::matrix_opt a_opt(a.csc());
  auto b = util::make_values<T>(n);

  std::vector<T> x = b;
  spblas::operation_state state;
  auto t = spblas::view::triangle(a_opt, spblas::view::uplo::upper);
  spblas::triangular_solve(util::test_policy{}, state, t, T(2), x, x);

  std::vector<T> two_b(n);
  for (std::size_t i = 0; i < n; i++) {
    two_b[i] = 2 * b[i];
  }
  expect_close(two_b, a.apply(x, spblas::view::uplo::upper,
                              spblas::view::diag::explicit_diagonal));
}

TEST(TriangularSolve, TransposedMatrixOpt) {
  using T = double;
  using I = spblas::index_t;

  auto [m, n, nnz] = util::square_dims[1];
  triangular_test_matrix<T, I> a(n, nnz);
  spblas::matrix_opt a_opt(a.csr());
  auto b = util::make_values<T>(n);
  std::vector<T> x(n);

  // upper(A^T) x = b, with A's matrix_opt attached.
  spblas::operation_state state;
  spblas::triangular_solve(
      state,
      spblas::view::triangle(spblas::view::transpose(a_opt),
                             spblas::view::uplo::upper),
      b, x);

  // r = upper(A^T) * x, i.e. r_i = sum_{j >= i} A_ji x_j.
  auto dense = util::to_dense(a.csr());
  std::vector<T> r(n, T(0));
  for (std::size_t i = 0; i < n; i++) {
    for (std::size_t j = i; j < n; j++) {
      r[i] += dense[j * n + i] * x[j];
    }
  }
  expect_close(b, r);
}

TEST(TriangularSolve, Int64Index) {
  using T = double;
  using I = std::int64_t;

  auto [m, n, nnz] = util::square_dims[1];
  triangular_test_matrix<T, I> a(n, nnz);
  auto b = util::make_values<T>(n);

  for (int csc = 0; csc < 2; csc++) {
    std::vector<T> x(n);
    spblas::operation_state state;
    if (csc) {
      spblas::triangular_solve(
          state, spblas::view::triangle(a.csc(), spblas::view::uplo::lower), b,
          x);
    } else {
      spblas::triangular_solve(
          state, spblas::view::triangle(a.csr(), spblas::view::uplo::lower), b,
          x);
    }
    expect_close(b, a.apply(x, spblas::view::uplo::lower,
                            spblas::view::diag::explicit_diagonal));
  }
}

TEST(TriangularSolve, Conjugated) {
  using T = std::complex<double>;
  using I = spblas::index_t;

  auto [m, n, nnz] = util::square_dims[1];
  triangular_test_matrix<double, I> real_a(n, nnz);

  // A complex matrix with the same pattern and a dominant diagonal.
  std::vector<T> values(real_a.values.size());
  for (std::size_t p = 0; p < values.size(); p++) {
    values[p] = T(real_a.values[p], 0.25 * real_a.values[p]);
  }
  spblas::csr_wrapper<T, I> a(values, real_a.rowptr, real_a.colind,
                              {I(n), I(n)}, I(values.size()));
  spblas::matrix_opt a_opt(a);

  std::vector<T> b(n);
  for (std::size_t i = 0; i < n; i++) {
    b[i] = T(double(i % 3) + 1, double(i % 2));
  }
  T alpha(0, 2);

  for (int use_opt = 0; use_opt < 2; use_opt++) {
    std::vector<T> x(n);
    spblas::operation_state state;
    if (use_opt) {
      auto t = spblas::view::triangle(spblas::view::conjugate(a_opt),
                                      spblas::view::uplo::lower);
      spblas::triangular_solve_inspect(state, t, alpha, b, x);
      spblas::triangular_solve(state, t, alpha, b, x);
    } else {
      spblas::triangular_solve(
          state,
          spblas::view::triangle(spblas::view::conjugate(a),
                                 spblas::view::uplo::upper),
          alpha, b, x);
    }

    // Residual: tri(conj(A)) x == alpha b
    bool lower = use_opt != 0;
    for (std::size_t i = 0; i < n; i++) {
      T r = 0;
      for (auto p = real_a.rowptr[i]; p < real_a.rowptr[i + 1]; p++) {
        std::size_t j = real_a.colind[p];
        if (j == i || (lower ? j < i : j > i)) {
          r += std::conj(values[p]) * x[j];
        }
      }
      EXPECT_NEAR(std::abs(r - alpha * b[i]), 0.0, 1e-10 * std::abs(b[i]));
    }
  }
}
