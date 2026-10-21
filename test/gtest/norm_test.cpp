#include <gtest/gtest.h>

#include <cmath>
#include <complex>
#include <limits>

#include "util.hpp"
#include <spblas/spblas.hpp>

namespace {

// Norms of a dense row-major matrix.
template <typename T>
auto dense_inf_norm(const std::vector<T>& d, std::size_t m, std::size_t n) {
  decltype(std::abs(T{})) norm = 0;
  for (std::size_t i = 0; i < m; i++) {
    decltype(norm) row_sum = 0;
    for (std::size_t j = 0; j < n; j++) {
      row_sum += std::abs(d[i * n + j]);
    }
    norm = std::max(norm, row_sum);
  }
  return norm;
}

template <typename T>
auto dense_frob_norm(const std::vector<T>& d) {
  long double sum = 0;
  for (auto&& v : d) {
    sum += static_cast<long double>(std::norm(v));
  }
  return static_cast<decltype(std::abs(T{}))>(std::sqrt(sum));
}

template <typename T>
void expect_norm_near(T expected, T actual) {
  EXPECT_NEAR(expected, actual,
              64 * std::numeric_limits<T>::epsilon() * expected);
}

} // namespace

TEST(Norm, CsrCscMatrixOptAndTranspose) {
  using T = double;
  using I = spblas::index_t;

  for (auto&& [m, n, nnz] : util::dims) {
    auto [values, rowptr, colind, shape, _] =
        spblas::generate_csr<T, I>(m, n, nnz);
    spblas::csr_wrapper<T, I> a(values, rowptr, colind, shape, nnz);
    auto dense = util::to_dense(a);

    // The same matrix in CSC.
    std::vector<T> t_values(nnz);
    std::vector<I> t_rowptr(n + 1), t_colind(nnz);
    spblas::csr_wrapper<T, I> a_t(t_values, t_rowptr, t_colind, {I(n), I(m)},
                                  nnz);
    spblas::operation_state state;
    spblas::transpose(state, a, a_t);
    spblas::csc_wrapper<T, I> a_csc(t_values, t_rowptr, t_colind, {I(m), I(n)},
                                    nnz);
    spblas::matrix_opt a_opt(a);
    spblas::matrix_opt a_csc_opt(a_csc);

    auto inf = dense_inf_norm(dense, m, n);
    auto frob = dense_frob_norm(dense);

    for (auto actual :
         {spblas::matrix_inf_norm(state, a),
          spblas::matrix_inf_norm(state, a_csc),
          spblas::matrix_inf_norm(state, a_opt),
          spblas::matrix_inf_norm(state, a_csc_opt),
          spblas::matrix_inf_norm(util::test_policy{}, state, a)}) {
      expect_norm_near(inf, actual);
    }
    for (auto actual :
         {spblas::matrix_frob_norm(state, a),
          spblas::matrix_frob_norm(state, a_csc),
          spblas::matrix_frob_norm(state, a_csc_opt),
          spblas::matrix_frob_norm(util::test_policy{}, state, a_opt)}) {
      expect_norm_near(frob, actual);
    }

    // inf-norm of the transpose is the 1-norm (max column sum) of A.
    double one_norm = 0;
    for (std::size_t j = 0; j < n; j++) {
      double column_sum = 0;
      for (std::size_t i = 0; i < m; i++) {
        column_sum += std::abs(dense[i * n + j]);
      }
      one_norm = std::max(one_norm, column_sum);
    }
    expect_norm_near(
        one_norm, spblas::matrix_inf_norm(state, spblas::view::transpose(a)));
    // The Frobenius norm does not change under transposition.
    expect_norm_near(
        frob, spblas::matrix_frob_norm(state, spblas::view::transpose(a)));
  }
}

TEST(Norm, ComplexAndConjugated) {
  using T = std::complex<float>;
  using I = spblas::index_t;

  for (auto&& [m, n, nnz] : util::dims) {
    auto [values, rowptr, colind, shape, _] =
        spblas::generate_csr<T, I>(m, n, nnz);
    spblas::csr_wrapper<T, I> a(values, rowptr, colind, shape, nnz);
    auto dense = util::to_dense(a);

    spblas::operation_state state;
    static_assert(
        std::is_same_v<decltype(spblas::matrix_frob_norm(state, a)), float>);

    expect_norm_near(dense_inf_norm(dense, m, n),
                     spblas::matrix_inf_norm(state, a));
    expect_norm_near(dense_frob_norm(dense),
                     spblas::matrix_frob_norm(state, a));
    expect_norm_near(
        dense_frob_norm(dense),
        spblas::matrix_frob_norm(state, spblas::view::conjugate(a)));
  }
}

TEST(Norm, FrobeniusDoesNotOverflowOrUnderflow) {
  using T = double;
  using I = spblas::index_t;

  for (double magnitude : {1e300, 1e-300}) {
    std::vector<T> values{3 * magnitude, -4 * magnitude, 12 * magnitude};
    std::vector<I> rowptr{0, 2, 3};
    std::vector<I> colind{0, 1, 1};
    spblas::csr_wrapper<T, I> a(values, rowptr, colind, {2, 2}, 3);

    spblas::operation_state state;
    // sqrt(9 + 16 + 144) = 13
    expect_norm_near(13 * magnitude, spblas::matrix_frob_norm(state, a));
    expect_norm_near(12 * magnitude, spblas::matrix_inf_norm(state, a));
  }
}

TEST(Norm, EmptyMatrix) {
  using T = float;
  using I = spblas::index_t;

  std::vector<T> values;
  std::vector<I> rowptr(4, 0), colind;
  spblas::csr_wrapper<T, I> a(values, rowptr, colind, {3, 3}, 0);

  spblas::operation_state state;
  EXPECT_EQ(spblas::matrix_inf_norm(state, a), 0.0f);
  EXPECT_EQ(spblas::matrix_frob_norm(state, a), 0.0f);
}

TEST(Norm, TransposedMatrixOpt) {
  using T = double;
  using I = spblas::index_t;

  for (auto&& [m, n, nnz] : util::dims) {
    auto [values, rowptr, colind, shape, _] =
        spblas::generate_csr<T, I>(m, n, nnz);
    spblas::csr_wrapper<T, I> a(values, rowptr, colind, shape, nnz);
    spblas::matrix_opt a_opt(a);

    spblas::operation_state state;
    // inf-norm of A^T is the 1-norm of A, for a matrix_opt as for a wrapper.
    expect_norm_near(
        spblas::matrix_inf_norm(state, spblas::view::transpose(a)),
        spblas::matrix_inf_norm(state, spblas::view::transpose(a_opt)));
    expect_norm_near(
        spblas::matrix_frob_norm(state, a),
        spblas::matrix_frob_norm(state, spblas::view::transpose(a_opt)));
  }
}
