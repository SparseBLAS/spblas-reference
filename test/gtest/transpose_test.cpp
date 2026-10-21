#include <gtest/gtest.h>

#include "util.hpp"
#include <spblas/spblas.hpp>

namespace {

template <typename T, typename A>
void transpose_test(A&& a, const std::vector<T>& a_dense, std::size_t m,
                    std::size_t n, std::size_t nnz) {
  using I = spblas::index_t;

  std::vector<T> b_values(nnz);
  std::vector<I> b_rowptr(n + 1);
  std::vector<I> b_colind(nnz);
  spblas::csr_wrapper<T, I> b(b_values, b_rowptr, b_colind, {I(n), I(m)},
                              I(nnz));

  spblas::operation_state state;
  spblas::transpose_inspect(state, a, b);
  spblas::transpose(state, a, b);

  EXPECT_EQ(b.size(), nnz);
  auto b_dense = util::to_dense(b);
  for (std::size_t i = 0; i < m; i++) {
    for (std::size_t j = 0; j < n; j++) {
      EXPECT_EQ(a_dense[i * n + j], b_dense[j * m + i]);
    }
  }
}

} // namespace

TEST(Transpose, Csr) {
  using T = float;
  using I = spblas::index_t;

  for (auto&& [m, n, nnz] : util::dims) {
    auto [values, rowptr, colind, shape, _] =
        spblas::generate_csr<T, I>(m, n, nnz);
    spblas::csr_wrapper<T, I> a(values, rowptr, colind, shape, nnz);

    transpose_test<T>(a, util::to_dense(a), m, n, nnz);
  }
}

TEST(Transpose, MatrixOptAndPolicy) {
  using T = float;
  using I = spblas::index_t;

  auto [m, n, nnz] = util::dims[1];
  auto [values, rowptr, colind, shape, _] =
      spblas::generate_csr<T, I>(m, n, nnz);
  spblas::csr_wrapper<T, I> a(values, rowptr, colind, shape, nnz);
  spblas::matrix_opt a_opt(a);

  transpose_test<T>(a_opt, util::to_dense(a), m, n, nnz);

  std::vector<T> b_values(nnz);
  std::vector<I> b_rowptr(n + 1);
  std::vector<I> b_colind(nnz);
  spblas::csr_wrapper<T, I> b(b_values, b_rowptr, b_colind, {I(n), I(m)},
                              I(nnz));
  spblas::operation_state state;
  spblas::transpose_inspect(util::test_policy{}, state, a, b);
  spblas::transpose(util::test_policy{}, state, a, b);
  EXPECT_EQ(util::to_dense(spblas::view::transpose(a)), util::to_dense(b));
}
