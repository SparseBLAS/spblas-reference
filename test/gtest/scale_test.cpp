#include <gtest/gtest.h>

#include <complex>

#include "util.hpp"
#include <spblas/spblas.hpp>

TEST(Scale, CsrCscMatrixOpt) {
  using T = float;
  using I = spblas::index_t;

  for (auto&& [m, n, nnz] : util::dims) {
    auto [values, rowptr, colind, shape, _] =
        spblas::generate_csr<T, I>(m, n, nnz);
    const auto original = values;

    spblas::operation_state state;
    spblas::csr_wrapper<T, I> a(values, rowptr, colind, shape, nnz);
    spblas::scale(state, T(2), a);
    for (std::size_t i = 0; i < values.size(); i++) {
      EXPECT_EQ(values[i], T(2) * original[i]);
    }

    spblas::csc_wrapper<T, I> a_csc(values, rowptr, colind, {I(n), I(m)}, nnz);
    spblas::scale(util::test_policy{}, state, T(0.5), a_csc);
    for (std::size_t i = 0; i < values.size(); i++) {
      EXPECT_EQ(values[i], original[i]);
    }

    spblas::matrix_opt a_opt(a);
    spblas::scale(state, -3, a_opt);
    for (std::size_t i = 0; i < values.size(); i++) {
      EXPECT_EQ(values[i], T(-3) * original[i]);
    }
  }
}

TEST(Scale, Complex) {
  using T = std::complex<double>;
  using I = spblas::index_t;

  auto [m, n, nnz] = util::dims[2];
  auto [values, rowptr, colind, shape, _] =
      spblas::generate_csr<T, I>(m, n, nnz);
  const auto original = values;

  spblas::csr_wrapper<T, I> a(values, rowptr, colind, shape, nnz);
  spblas::operation_state state;
  spblas::scale(state, T(0, 1), a);
  for (std::size_t i = 0; i < values.size(); i++) {
    EXPECT_EQ(values[i], T(0, 1) * original[i]);
  }
}
