#include <gtest/gtest.h>

#include <complex>

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
  if (b.sort_state() == spblas::sort_state::sorted) {
    EXPECT_TRUE(util::sorted_slices(b_rowptr, b_colind));
  }
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

TEST(Transpose, ConjugateTranspose) {
  using T = std::complex<double>;
  using I = spblas::index_t;

  for (auto&& [m, n, nnz] : util::dims) {
    auto [values, rowptr, colind, shape, _] =
        spblas::generate_csr<T, I>(m, n, nnz);
    spblas::csr_wrapper<T, I> a(values, rowptr, colind, shape, nnz);
    auto a_dense = util::to_dense(a);
    spblas::matrix_opt a_opt(a);

    for (int use_opt = 0; use_opt < 2; use_opt++) {
      std::vector<T> b_values(nnz);
      std::vector<I> b_rowptr(n + 1), b_colind(nnz);
      spblas::csr_wrapper<T, I> b(b_values, b_rowptr, b_colind, {I(n), I(m)},
                                  nnz);

      spblas::operation_state state;
      if (use_opt) {
        spblas::transpose(state, spblas::view::conjugate(a_opt), b);
      } else {
        spblas::transpose_inspect(state, spblas::view::conjugate(a), b);
        spblas::transpose(state, spblas::view::conjugate(a), b);
      }

      // B = A^H
      auto b_dense = util::to_dense(b);
      for (std::size_t i = 0; i < m; i++) {
        for (std::size_t j = 0; j < n; j++) {
          EXPECT_EQ(std::conj(a_dense[i * n + j]), b_dense[j * m + i]);
        }
      }
    }
  }
}

TEST(Transpose, ConjugateOfRealIsTranspose) {
  using T = float;
  using I = spblas::index_t;

  auto [m, n, nnz] = util::dims[0];
  auto [values, rowptr, colind, shape, _] =
      spblas::generate_csr<T, I>(m, n, nnz);
  spblas::csr_wrapper<T, I> a(values, rowptr, colind, shape, nnz);

  transpose_test<T>(spblas::view::conjugate(a), util::to_dense(a), m, n, nnz);
}

TEST(Transpose, CscInput) {
  using T = float;
  using I = spblas::index_t;

  for (auto&& [m, n, nnz] : util::dims) {
    auto [values, colptr, rowind, shape, _] =
        spblas::generate_csc<T, I>(m, n, nnz);
    spblas::csc_wrapper<T, I> a(values, colptr, rowind, shape, nnz);

    transpose_test<T>(a, util::to_dense(a), m, n, nnz);
  }
}

// B's sort_state() says what the input order guarantees.
TEST(Transpose, OutputSortState) {
  using T = float;
  using I = spblas::index_t;
  using spblas::sort_state;

  auto [m, n, nnz] = util::dims[1];
  std::vector<T> b_values(nnz);
  std::vector<I> b_rowptr(n + 1), b_colind(nnz);
  spblas::csr_wrapper<T, I> b(b_values, b_rowptr, b_colind, {I(n), I(m)},
                              I(nnz));
  spblas::operation_state state;

  // CSR input: visited row by row, so B's rows are sorted.
  auto [values, rowptr, colind, shape, _] =
      spblas::generate_csr<T, I>(m, n, nnz);
  util::shuffle_slices(rowptr, colind, values);
  spblas::csr_wrapper<T, I> a(values, rowptr, colind, shape, nnz);
  b.set_sort_state(sort_state::unsorted);
  spblas::transpose(state, a, b);
  EXPECT_EQ(b.sort_state(), sort_state::sorted);
  EXPECT_TRUE(util::sorted_slices(b_rowptr, b_colind));

  // CSC input of unknown order: unknown (B's earlier claim is replaced).
  auto [c_values, colptr, rowind, c_shape, c_nnz] =
      spblas::generate_csc<T, I>(m, n, nnz);
  spblas::csc_wrapper<T, I> c(c_values, colptr, rowind, c_shape, c_nnz);
  spblas::transpose(state, spblas::matrix_opt(c), b);
  EXPECT_EQ(b.sort_state(), sort_state::unknown);

  // CSC input known to be sorted: B's rows are its sorted columns.
  spblas::sort_matrix(state, c);
  spblas::transpose(state, c, b);
  EXPECT_EQ(b.sort_state(), sort_state::sorted);
  EXPECT_TRUE(util::sorted_slices(b_rowptr, b_colind));
}

TEST(Transpose, Errors) {
  using T = float;
  using I = spblas::index_t;

  auto [values, rowptr, colind, shape, nnz] =
      spblas::generate_csr<T, I>(10, 20, 30);
  spblas::csr_wrapper<T, I> a(values, rowptr, colind, shape, nnz);
  spblas::operation_state state;

  std::vector<T> b_values(nnz);
  std::vector<I> b_rowptr(21), b_colind(nnz), short_rowptr(20);
  std::vector<T> small_values(nnz - 1);

  spblas::csr_wrapper<T, I> wrong_shape(b_values, b_rowptr, b_colind,
                                        {I(10), I(20)}, nnz);
  EXPECT_THROW(spblas::transpose(state, a, wrong_shape), std::invalid_argument);

  spblas::csr_wrapper<T, I> bad_rowptr(b_values, short_rowptr, b_colind,
                                       {I(20), I(10)}, nnz);
  EXPECT_THROW(spblas::transpose(state, a, bad_rowptr), std::invalid_argument);

  spblas::csr_wrapper<T, I> too_small(small_values, b_rowptr, b_colind,
                                      {I(20), I(10)}, nnz);
  EXPECT_THROW(spblas::transpose(state, a, too_small), std::runtime_error);
}

TEST(Transpose, OfTransposedMatrixOpt) {
  using T = float;
  using I = spblas::index_t;

  auto [m, n, nnz] = util::dims[1];
  auto [values, rowptr, colind, shape, _] =
      spblas::generate_csr<T, I>(m, n, nnz);
  spblas::csr_wrapper<T, I> a(values, rowptr, colind, shape, nnz);
  spblas::matrix_opt a_opt(a);

  // B = (A^T)^T = A
  std::vector<T> b_values(nnz);
  std::vector<I> b_rowptr(m + 1), b_colind(nnz);
  spblas::csr_wrapper<T, I> b(b_values, b_rowptr, b_colind, {I(m), I(n)}, nnz);
  spblas::operation_state state;
  spblas::transpose(state, spblas::view::transpose(a_opt), b);

  EXPECT_EQ(util::to_dense(a), util::to_dense(b));
}
