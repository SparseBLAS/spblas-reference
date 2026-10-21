#include <gtest/gtest.h>

#include "util.hpp"
#include <spblas/spblas.hpp>

TEST(Add, CsrCsrCsr) {
  using T = float;
  using I = spblas::index_t;

  for (auto&& [m, n, nnz] : util::dims) {
    auto [a_values, a_rowptr, a_colind, a_shape, a_nnz] =
        spblas::generate_csr<T, I>(m, n, nnz, 0);
    auto [b_values, b_rowptr, b_colind, b_shape, b_nnz] =
        spblas::generate_csr<T, I>(m, n, nnz, 1);

    spblas::csr_wrapper<T, I> a(a_values, a_rowptr, a_colind, a_shape, a_nnz);
    spblas::csr_wrapper<T, I> b(b_values, b_rowptr, b_colind, b_shape, b_nnz);
    spblas::csr_wrapper<T, I> c(m, n);

    spblas::operation_state state;
    spblas::add_inspect(state, a, b, c);
    spblas::add_compute(state, a, b, c);

    std::vector<T> c_values(state.get_result_nnz());
    std::vector<I> c_colind(state.get_result_nnz());
    std::vector<I> c_rowptr(m + 1);
    c.update(c_values, c_rowptr, c_colind);

    spblas::add_fill(state, a, b, c);

    // Pattern is the union of both patterns; values are A + B.
    auto a_dense = util::to_dense(a);
    auto b_dense = util::to_dense(b);
    std::size_t union_positions = 0;
    for (std::size_t i = 0; i < a_dense.size(); i++) {
      union_positions += (a_dense[i] != T(0) || b_dense[i] != T(0)) ? 1 : 0;
    }
    EXPECT_EQ(state.get_result_nnz(), util::count_positions(c));
    EXPECT_EQ(c_rowptr.back(), state.get_result_nnz());

    auto c_dense = util::to_dense(c);
    for (std::size_t i = 0; i < c_dense.size(); i++) {
      EXPECT_EQ_(a_dense[i] + b_dense[i], c_dense[i]);
    }
    // Generated values are nonzero, so value positions equal the pattern.
    EXPECT_EQ(state.get_result_nnz(), union_positions);
  }
}

TEST(Add, MatrixOptAndPolicy) {
  using T = double;
  using I = spblas::index_t;

  auto [m, n, nnz] = util::dims[2];
  auto [a_values, a_rowptr, a_colind, a_shape, a_nnz] =
      spblas::generate_csr<T, I>(m, n, nnz, 0);
  auto [b_values, b_rowptr, b_colind, b_shape, b_nnz] =
      spblas::generate_csr<T, I>(m, n, nnz, 1);

  spblas::matrix_opt a(
      spblas::csr_wrapper<T, I>(a_values, a_rowptr, a_colind, a_shape, a_nnz));
  spblas::matrix_opt b(
      spblas::csr_wrapper<T, I>(b_values, b_rowptr, b_colind, b_shape, b_nnz));
  spblas::csr_wrapper<T, I> c(m, n);

  spblas::operation_state state;
  spblas::add_inspect(util::test_policy{}, state, a, b, c);
  spblas::add_compute(util::test_policy{}, state, a, b, c);

  std::vector<T> c_values(state.get_result_nnz());
  std::vector<I> c_colind(state.get_result_nnz());
  std::vector<I> c_rowptr(m + 1);
  c.update(c_values, c_rowptr, c_colind);

  spblas::add_fill(util::test_policy{}, state, a, b, c);

  auto a_dense = util::to_dense(a.base());
  auto b_dense = util::to_dense(b.base());
  auto c_dense = util::to_dense(c);
  for (std::size_t i = 0; i < c_dense.size(); i++) {
    EXPECT_EQ_(a_dense[i] + b_dense[i], c_dense[i]);
  }
}
