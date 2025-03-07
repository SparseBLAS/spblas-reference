#include <gtest/gtest.h>

#include "util.hpp"
#include <spblas/backend/spa_accumulator.hpp>
#include <spblas/spblas.hpp>

#include <fmt/core.h>
#include <fmt/ranges.h>

TEST(MixedViews, SpGEMM_CsrCsc) {
  using T = float;
  using I = spblas::index_t;

  for (auto&& [m, k, nnz] : util::dims) {
    for (auto&& n : {m, k}) {
      auto [a_values, a_rowptr, a_colind, a_shape, a_nnz] =
          spblas::generate_csr<T, I>(m, k, nnz);

      auto [b_values, b_rowptr, b_colind, b_shape, b_nnz] =
          spblas::generate_csr<T, I>(k, n, nnz);

      // We will be multiplying a times b.
      spblas::csr_view<T, I> a(a_values, a_rowptr, a_colind, a_shape, a_nnz);
      spblas::csr_view<T, I> b(b_values, b_rowptr, b_colind, b_shape, b_nnz);

      // But we'd like the second operand to be a CSC matrix.
      // We first transpose b.

      std::vector<T> b_t_values(b.size());
      std::vector<I> b_t_rowptr(b.shape()[1] + 1);
      std::vector<I> b_t_colind(b.size());

      spblas::csr_view<T, I> b_t(b_t_values, b_t_rowptr, b_t_colind,
                                 {b.shape()[1], b.shape()[0]}, 0);

      spblas::transpose(b, b_t);

      // We then build a CSC representation of b from b_t.
      spblas::csc_view<T, I> b_csc(b_t.values(), b_t.rowptr(), b_t.colind(),
                                   {b_t.shape()[1], b_t.shape()[0]},
                                   b_t.size());

      // Now let's multiply a * b_csc -> c.

      std::vector<I> c_rowptr(m + 1);
      spblas::csr_view<T, I> c(nullptr, c_rowptr.data(), nullptr, {m, n}, 0);

      auto info = spblas::multiply_compute(a, b_csc, c);

      std::vector<T> c_values(info.result_nnz());
      std::vector<I> c_colind(info.result_nnz());

      c.update(c_values, c_rowptr, c_colind);

      spblas::multiply_fill(info, a, b_csc, c);

      // Now that we have c, let's compute a reference c_ref.
      // We perform a * b -> c_ref

      std::vector<I> c_ref_rowptr(m + 1);

      spblas::csr_view<T, I> c_ref(nullptr, c_ref_rowptr.data(), nullptr,
                                   {m, n}, 0);

      info = spblas::multiply_compute(a, b, c_ref);

      std::vector<T> c_ref_values(info.result_nnz());
      std::vector<I> c_ref_colind(info.result_nnz());

      c_ref.update(c_ref_values, c_ref_rowptr, c_ref_colind);

      spblas::multiply_fill(info, a, b, c_ref);

      spblas::__backend::spa_accumulator<T, I> c_row_acc(c.shape()[1]);

      for (auto&& [i, c_row] : spblas::__backend::rows(c)) {
        c_row_acc.clear();

        auto&& c_ref_row = spblas::__backend::lookup_row(c_ref, i);

        EXPECT_EQ(c_row.size(), c_ref_row.size());

        for (auto&& [j, v] : c_row) {
          c_row_acc[j] += v;
        }

        for (auto&& [j, v] : c_ref_row) {
          EXPECT_EQ_(v, c_row_acc[j]);
        }
      }
    }
  }
}

TEST(MixedViews, SpGEMM_CscCsr) {
  using T = float;
  using I = spblas::index_t;

  for (auto&& [m, k, nnz] : util::dims) {
    for (auto&& n : {m, k}) {
      auto [a_values, a_rowptr, a_colind, a_shape, a_nnz] =
          spblas::generate_csr<T, I>(m, k, nnz);

      auto [b_values, b_rowptr, b_colind, b_shape, b_nnz] =
          spblas::generate_csr<T, I>(k, n, nnz);

      // We will be multiplying a times b.
      spblas::csr_view<T, I> a(a_values, a_rowptr, a_colind, a_shape, a_nnz);
      spblas::csr_view<T, I> b(b_values, b_rowptr, b_colind, b_shape, b_nnz);

      // But we'd like the first operand to be a CSC matrix.
      // We first transpose a.

      std::vector<T> a_t_values(a.size());
      std::vector<I> a_t_rowptr(a.shape()[1] + 1);
      std::vector<I> a_t_colind(a.size());

      spblas::csr_view<T, I> a_t(a_t_values, a_t_rowptr, a_t_colind,
                                 {a.shape()[1], a.shape()[0]}, 0);

      spblas::transpose(a, a_t);

      // We then build a CSC representation of a from b_t.
      spblas::csc_view<T, I> a_csc(a_t.values(), a_t.rowptr(), a_t.colind(),
                                   {a_t.shape()[1], a_t.shape()[0]},
                                   a_t.size());

      // Now let's multiply a_csc * b -> c.

      std::vector<I> c_rowptr(m + 1);
      spblas::csr_view<T, I> c(nullptr, c_rowptr.data(), nullptr, {m, n}, 0);

      auto info = spblas::multiply_compute(a_csc, b, c);

      std::vector<T> c_values(info.result_nnz());
      std::vector<I> c_colind(info.result_nnz());

      c.update(c_values, c_rowptr, c_colind);

      spblas::multiply_fill(info, a_csc, b, c);

      // Now that we have c, let's compute a reference c_ref.
      // We perform a * b -> c_ref

      std::vector<I> c_ref_rowptr(m + 1);

      spblas::csr_view<T, I> c_ref(nullptr, c_ref_rowptr.data(), nullptr,
                                   {m, n}, 0);

      info = spblas::multiply_compute(a, b, c_ref);

      std::vector<T> c_ref_values(info.result_nnz());
      std::vector<I> c_ref_colind(info.result_nnz());

      c_ref.update(c_ref_values, c_ref_rowptr, c_ref_colind);

      spblas::multiply_fill(info, a, b, c_ref);

      spblas::__backend::spa_accumulator<T, I> c_row_acc(c.shape()[1]);

      for (auto&& [i, c_row] : spblas::__backend::rows(c)) {
        c_row_acc.clear();

        auto&& c_ref_row = spblas::__backend::lookup_row(c_ref, i);

        EXPECT_EQ(c_row.size(), c_ref_row.size());

        for (auto&& [j, v] : c_row) {
          c_row_acc[j] += v;
        }

        for (auto&& [j, v] : c_ref_row) {
          EXPECT_EQ_(v, c_row_acc[j]);
        }
      }
    }
  }
}
