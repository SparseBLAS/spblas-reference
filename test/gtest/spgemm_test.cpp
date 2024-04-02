#include <gtest/gtest.h>

#include "util.hpp"
#include <spblas/backend/spa_accumulator.hpp>
#include <spblas/spblas.hpp>

#include <fmt/core.h>
#include <fmt/ranges.h>

TEST(CsrView, SpGEMM) {
  using T = float;
  using I = spblas::index_t;

  for (auto&& [m, k, nnz] : util::dims) {
    for (auto&& n : {m, k}) {
      auto [a_values, a_rowptr, a_colind, a_shape, a_nnz] =
          spblas::generate_csr<T, I>(m, k, nnz);

      auto [b_values, b_rowptr, b_colind, b_shape, b_nnz] =
          spblas::generate_csr<T, I>(k, n, nnz);

      spblas::csr_view<T, I> a(a_values, a_rowptr, a_colind, a_shape, a_nnz);
      spblas::csr_view<T, I> b(b_values, b_rowptr, b_colind, b_shape, b_nnz);

      std::vector<I> c_rowptr(m + 1);

      spblas::csr_view<T, I> c(nullptr, c_rowptr.data(), nullptr, {m, n}, 0);

      auto info = spblas::multiply_inspect(a, b, c);

      std::vector<T> c_values(info.result_nnz());
      std::vector<I> c_colind(info.result_nnz());

      c.update(c_values, c_rowptr, c_colind);

      spblas::multiply_execute(info, a, b, c);

      spblas::__backend::spa_accumulator<T, I> c_row_ref(
          spblas::__backend::shape(c)[1]);

      spblas::__backend::spa_accumulator<T, I> c_row_acc(
          spblas::__backend::shape(c)[1]);

      for (auto&& [i, a_row] : spblas::__backend::rows(a)) {
        c_row_ref.clear();
        for (auto&& [k, a_v] : a_row) {
          auto&& b_row = spblas::__backend::lookup_row(b, k);

          for (auto&& [j, b_v] : b_row) {
            c_row_ref[j] += a_v * b_v;
          }
        }

        auto&& c_row = spblas::__backend::lookup_row(c, i);

        // Accumulate output into `c_row_acc` so that we can allow
        // duplicate column indices.
        c_row_acc.clear();
        for (auto&& [j, c_v] : c_row) {
          c_row_acc[j] += c_v;
        }

        for (auto&& [j, c_v] : c_row) {
          EXPECT_EQ_(c_row_ref[j], c_row_acc[j]);
        }

        EXPECT_EQ(c_row_ref.size(), c_row_acc.size());
      }
    }
  }
}

TEST(CsrView, SpGEMM_AScaled) {
  using T = float;
  using I = spblas::index_t;

  T alpha = 2.0f;

  for (auto&& [m, k, nnz] : util::dims) {
    for (auto&& n : {m, k}) {
      auto [a_values, a_rowptr, a_colind, a_shape, a_nnz] =
          spblas::generate_csr<T, I>(m, k, nnz);

      auto [b_values, b_rowptr, b_colind, b_shape, b_nnz] =
          spblas::generate_csr<T, I>(k, n, nnz);

      spblas::csr_view<T, I> a(a_values, a_rowptr, a_colind, a_shape, a_nnz);
      spblas::csr_view<T, I> b(b_values, b_rowptr, b_colind, b_shape, b_nnz);

      std::vector<I> c_rowptr(m + 1);

      spblas::csr_view<T, I> c(nullptr, c_rowptr.data(), nullptr, {m, n}, 0);

      auto info = spblas::multiply_inspect(a, b, c);

      std::vector<T> c_values(info.result_nnz());
      std::vector<I> c_colind(info.result_nnz());

      c.update(c_values, c_rowptr, c_colind);

      spblas::multiply_execute(info, spblas::scaled(alpha, a), b, c);

      spblas::__backend::spa_accumulator<T, I> c_row_ref(
          spblas::__backend::shape(c)[1]);

      spblas::__backend::spa_accumulator<T, I> c_row_acc(
          spblas::__backend::shape(c)[1]);

      for (auto&& [i, a_row] : spblas::__backend::rows(a)) {
        c_row_ref.clear();
        for (auto&& [k, a_v] : a_row) {
          auto&& b_row = spblas::__backend::lookup_row(b, k);

          for (auto&& [j, b_v] : b_row) {
            c_row_ref[j] += alpha * a_v * b_v;
          }
        }

        auto&& c_row = spblas::__backend::lookup_row(c, i);

        // Accumulate output into `c_row_acc` so that we can allow
        // duplicate column indices.
        c_row_acc.clear();
        for (auto&& [j, c_v] : c_row) {
          c_row_acc[j] += c_v;
        }

        for (auto&& [j, c_v] : c_row) {
          EXPECT_EQ_(c_row_ref[j], c_row_acc[j]);
        }

        EXPECT_EQ(c_row_ref.size(), c_row_acc.size());
      }
    }
  }
}

TEST(CsrView, SpGEMM_BScaled) {
  using T = float;
  using I = spblas::index_t;

  T alpha = 2.0f;

  for (auto&& [m, k, nnz] : util::dims) {
    for (auto&& n : {m, k}) {
      auto [a_values, a_rowptr, a_colind, a_shape, a_nnz] =
          spblas::generate_csr<T, I>(m, k, nnz);

      auto [b_values, b_rowptr, b_colind, b_shape, b_nnz] =
          spblas::generate_csr<T, I>(k, n, nnz);

      spblas::csr_view<T, I> a(a_values, a_rowptr, a_colind, a_shape, a_nnz);
      spblas::csr_view<T, I> b(b_values, b_rowptr, b_colind, b_shape, b_nnz);

      std::vector<I> c_rowptr(m + 1);

      spblas::csr_view<T, I> c(nullptr, c_rowptr.data(), nullptr, {m, n}, 0);

      auto info = spblas::multiply_inspect(a, b, c);

      std::vector<T> c_values(info.result_nnz());
      std::vector<I> c_colind(info.result_nnz());

      c.update(c_values, c_rowptr, c_colind);

      spblas::multiply_execute(info, a, spblas::scaled(alpha, b), c);

      spblas::__backend::spa_accumulator<T, I> c_row_ref(
          spblas::__backend::shape(c)[1]);

      spblas::__backend::spa_accumulator<T, I> c_row_acc(
          spblas::__backend::shape(c)[1]);

      for (auto&& [i, a_row] : spblas::__backend::rows(a)) {
        c_row_ref.clear();
        for (auto&& [k, a_v] : a_row) {
          auto&& b_row = spblas::__backend::lookup_row(b, k);

          for (auto&& [j, b_v] : b_row) {
            c_row_ref[j] += a_v * alpha * b_v;
          }
        }

        auto&& c_row = spblas::__backend::lookup_row(c, i);

        // Accumulate output into `c_row_acc` so that we can allow
        // duplicate column indices.
        c_row_acc.clear();
        for (auto&& [j, c_v] : c_row) {
          c_row_acc[j] += c_v;
        }

        for (auto&& [j, c_v] : c_row) {
          EXPECT_EQ_(c_row_ref[j], c_row_acc[j]);
        }

        EXPECT_EQ(c_row_ref.size(), c_row_acc.size());
      }
    }
  }
}
