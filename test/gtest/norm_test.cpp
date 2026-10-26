#include <gtest/gtest.h>

#include <cmath>
#include <complex>
#include <limits>
#include <tuple>

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
auto dense_one_norm(const std::vector<T>& d, std::size_t m, std::size_t n) {
  decltype(std::abs(T{})) norm = 0;
  for (std::size_t j = 0; j < n; j++) {
    decltype(norm) column_sum = 0;
    for (std::size_t i = 0; i < m; i++) {
      column_sum += std::abs(d[i * n + j]);
    }
    norm = std::max(norm, column_sum);
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

    // The 1-norm (max column sum) is the inf-norm of the transpose, and the
    // inf-norm the 1-norm of the transpose.
    auto one = dense_one_norm(dense, m, n);
    for (auto actual :
         {spblas::matrix_one_norm(state, a),
          spblas::matrix_one_norm(state, a_csc),
          spblas::matrix_one_norm(state, a_opt),
          spblas::matrix_one_norm(state, a_csc_opt),
          spblas::matrix_one_norm(util::test_policy{}, state, a),
          spblas::matrix_inf_norm(state, spblas::view::transpose(a))}) {
      expect_norm_near(one, actual);
    }
    expect_norm_near(
        inf, spblas::matrix_one_norm(state, spblas::view::transpose(a_opt)));
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
    expect_norm_near(dense_one_norm(dense, m, n),
                     spblas::matrix_one_norm(state, a));
    expect_norm_near(
        dense_one_norm(dense, m, n),
        spblas::matrix_one_norm(state, spblas::view::conjugate(a)));
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

// NaN propagates to both norms, whatever the entry order; infinite entries
// give an infinite norm (not NaN), unless there is also a NaN.
TEST(Norm, NanAndInfinity) {
  using T = double;
  using I = spblas::index_t;
  const T nan = std::numeric_limits<T>::quiet_NaN();
  const T inf = std::numeric_limits<T>::infinity();

  // One entry per row of a k x 1 matrix.
  auto norms = [](std::vector<T> values) {
    std::vector<I> rowptr(values.size() + 1), colind(values.size(), 0);
    for (std::size_t i = 0; i < values.size(); i++) {
      rowptr[i + 1] = I(i + 1);
    }
    spblas::csr_wrapper<T, I> a(values, rowptr, colind,
                                {I(values.size()), I(1)}, I(values.size()));
    spblas::operation_state state;
    return std::tuple(spblas::matrix_inf_norm(state, a),
                      spblas::matrix_frob_norm(state, a),
                      spblas::matrix_one_norm(state, a));
  };

  for (auto&& values :
       {std::vector<T>{nan}, std::vector<T>{1, nan}, std::vector<T>{nan, 1},
        std::vector<T>{inf, nan}, std::vector<T>{nan, inf},
        std::vector<T>{inf, nan, 1}}) {
    auto [inf_norm, frob_norm, one_norm] = norms(values);
    EXPECT_TRUE(std::isnan(inf_norm));
    EXPECT_TRUE(std::isnan(frob_norm));
    EXPECT_TRUE(std::isnan(one_norm));
  }
  for (auto&& values :
       {std::vector<T>{inf}, std::vector<T>{1, -inf}, std::vector<T>{inf, -inf},
        std::vector<T>{inf, 2, inf}}) {
    auto [inf_norm, frob_norm, one_norm] = norms(values);
    EXPECT_EQ(inf_norm, inf);
    EXPECT_EQ(frob_norm, inf);
    EXPECT_EQ(one_norm, inf);
  }

  // A complex entry whose components are both infinite.
  using C = std::complex<T>;
  std::vector<C> values{C(inf, -inf)};
  std::vector<I> rowptr{0, 1}, colind{0};
  spblas::csr_wrapper<C, I> a(values, rowptr, colind, {1, 1}, 1);
  spblas::operation_state state;
  EXPECT_EQ(spblas::matrix_frob_norm(state, a), inf);
  EXPECT_EQ(spblas::matrix_inf_norm(state, a), inf);
  EXPECT_EQ(spblas::matrix_one_norm(state, a), inf);
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
  EXPECT_EQ(spblas::matrix_one_norm(state, a), 0.0f);
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
