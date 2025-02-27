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
    SCOPED_TRACE(m);
    SCOPED_TRACE(k);
    SCOPED_TRACE(nnz);
    auto [a_values, a_rowptr, a_colind, a_shape, a_nnz] =
        spblas::generate_csr<T, I>(m, k, nnz);
    spblas::index<> b_shape{a_shape[1], a_shape[0]};
    auto b_nnz = a_nnz;
    std::vector<O> b_rowptr(b_shape[0] + 1);
    std::vector<I> b_colind(a_nnz);
    std::vector<T> b_values(a_nnz);

    spblas::csr_view<T, I> a(a_values, a_rowptr, a_colind, a_shape, a_nnz);
    spblas::csr_view<T, I> b(b_values, b_rowptr, b_colind, b_shape, b_nnz);

    auto info = spblas::transpose_inspect(a, b);

    spblas::transpose(info, a, b);

    std::vector<I> a_rowind;
    std::vector<I> b_rowind;

    for (auto&& [i, row] : spblas::__backend::rows(a)) {
      for (auto&& [j, v] : row) {
        a_rowind.push_back(i);
      }
    }
    for (auto&& [i, row] : spblas::__backend::rows(b)) {
      for (auto&& [j, v] : row) {
        b_rowind.push_back(i);
      }
    }
    spblas::__ranges::sort(
        spblas::__ranges::views::zip(a_colind, a_rowind, a_values));
    // there may be duplicate values in rows, so we need to sort each row of b
    // by column index
    for (auto i : spblas::__ranges::views::iota(I{}, b_shape[0])) {
      const auto b_begin = b_rowptr[i];
      const auto b_end = b_rowptr[i + 1];
      const auto row_colind =
          std::span{b_colind.begin() + b_begin, b_colind.begin() + b_end};
      const auto row_values =
          std::span{b_values.begin() + b_begin, b_values.begin() + b_end};
      spblas::__ranges::sort(
          spblas::__ranges::views::zip(row_colind, row_values));
    }
    EXPECT_EQ(a_rowind, b_colind);
    EXPECT_EQ(a_colind, b_rowind);
    EXPECT_EQ(a_values, b_values);
  }
}