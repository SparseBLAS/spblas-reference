#include <gtest/gtest.h>

#include "util.hpp"
#include <spblas/algorithms/transpose.hpp>
#include <spblas/backend/spa_accumulator.hpp>
#include <spblas/spblas.hpp>

#include <fmt/core.h>
#include <fmt/ranges.h>

TEST(CsrView, Transpose) {
  using T = float;
  using I = spblas::index_t;
  using O = spblas::offset_t;

  for (auto&& [m, k, nnz] : util::dims) {
    // Generate CSR Matrix A.
    auto [a_values, a_rowptr, a_colind, a_shape, a_nnz] =
        spblas::generate_csr<T, I>(m, k, nnz);

    spblas::csr_view<T, I, O> a(a_values, a_rowptr, a_colind, a_shape, a_nnz);

    // Transpose; B = A_T

    spblas::index b_shape(a.shape()[1], a.shape()[0]);

    std::vector<O> b_rowptr(b_shape[0] + 1);
    std::vector<I> b_colind(a.size());
    std::vector<T> b_values(a.size());

    spblas::csr_view<T, I, O> b(b_values, b_rowptr, b_colind, b_shape,
                                a.size());

    auto info = spblas::transpose_inspect(a, b);
    spblas::transpose(info, a, b);

    // Create transposed COO for reference.
    std::vector<T> ref_values;
    std::vector<I> ref_rowind;
    std::vector<I> ref_colind;

    for (auto&& [i, row] : spblas::__backend::rows(a)) {
      for (auto&& [j, v] : row) {
        ref_values.push_back(v);
        ref_rowind.push_back(j);
        ref_colind.push_back(i);
      }
    }

    // Create COO from transposed matrix for test.
    std::vector<T> test_values;
    std::vector<T> test_rowind;
    std::vector<T> test_colind;

    for (auto&& [i, row] : spblas::__backend::rows(b)) {
      for (auto&& [j, v] : row) {
        test_values.push_back(v);
        test_rowind.push_back(i);
        test_colind.push_back(j);
      }
    }

    // Ensure both COO matrices are sorted.
    spblas::__ranges::sort(
        spblas::__ranges::views::zip(ref_rowind, ref_colind, ref_values));
    spblas::__ranges::sort(
        spblas::__ranges::views::zip(test_rowind, test_colind, test_values));

    EXPECT_EQ(ref_values.size(), test_values.size());
    EXPECT_EQ(ref_rowind.size(), test_rowind.size());
    EXPECT_EQ(ref_colind.size(), test_colind.size());

    for (auto&& [a, b] :
         spblas::__ranges::views::zip(ref_values, test_values)) {
      EXPECT_EQ_(a, b);
    }

    for (auto&& [a, b] :
         spblas::__ranges::views::zip(ref_rowind, test_rowind)) {
      EXPECT_EQ(a, b);
    }

    for (auto&& [a, b] :
         spblas::__ranges::views::zip(ref_colind, test_colind)) {
      EXPECT_EQ(a, b);
    }
  }
}
