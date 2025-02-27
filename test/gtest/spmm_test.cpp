#include <gtest/gtest.h>

#include "util.hpp"
#include <spblas/spblas.hpp>

TEST(CsrView, SpMM) {
  namespace md = spblas::__mdspan;

  using T = float;
  using I = spblas::index_t;

  for (auto [m, k, nnz] : util::dims) {
    for (auto n : {1, 8, 32, 64, 512}) {
      auto [values, rowptr, colind, shape, _] =
          spblas::generate_csr<T, I>(m, k, nnz);

      spblas::csr_view<T, I> a(values, rowptr, colind, shape, nnz);

      auto [b_values, b_shape] = spblas::generate_dense<T>(k, n);

      std::vector<T> c_values(m * n, 0);

      md::mdspan b(b_values.data(), k, n);
      md::mdspan c(c_values.data(), m, n);

      spblas::multiply(a, b, c);

      std::vector<T> c_ref(m * n, 0);

      for (I i = 0; i < m; i++) {
        for (I k_ptr = rowptr[i]; k_ptr < rowptr[i + 1]; k_ptr++) {
          I k = colind[k_ptr];
          T v = values[k_ptr];

          for (I j = 0; j < n; j++) {
            c_ref[i * n + j] += v * b_values[k * n + j];
          }
        }
      }

      for (I i = 0; i < c_ref.size(); i++) {
        EXPECT_EQ_(c_ref[i], c_values[i]);
      }
    }
  }
}

TEST(CsrView, SpMM_AScaled) {
  namespace md = spblas::__mdspan;

  using T = float;
  using I = spblas::index_t;

  T scaling_factor = 2.0f;

  for (auto&& [m, k, nnz] : util::dims) {
    for (auto&& n : {1, 8, 32, 64, 512}) {
      auto [values, rowptr, colind, shape, _] =
          spblas::generate_csr<T, I>(m, k, nnz);

      spblas::csr_view<T, I> a(values, rowptr, colind, shape, nnz);

      auto [b_values, b_shape] = spblas::generate_dense<T>(k, n);

      std::vector<T> c_values(m * n, 0);

      md::mdspan b(b_values.data(), k, n);
      md::mdspan c(c_values.data(), m, n);

      auto a_view = spblas::scaled(scaling_factor, a);

      spblas::multiply(a_view, b, c);

      std::vector<T> c_ref(m * n, 0);

      for (I i = 0; i < m; i++) {
        for (I k_ptr = rowptr[i]; k_ptr < rowptr[i + 1]; k_ptr++) {
          I k = colind[k_ptr];
          T v = values[k_ptr];

          for (I j = 0; j < n; j++) {
            c_ref[i * n + j] += scaling_factor * v * b_values[k * n + j];
          }
        }
      }

      for (I i = 0; i < c_ref.size(); i++) {
        EXPECT_EQ_(c_ref[i], c_values[i]);
      }
    }
  }
}

TEST(CsrView, SpMM_BScaled) {
  namespace md = spblas::__mdspan;

  using T = float;
  using I = spblas::index_t;

  T scaling_factor = 2.0f;

  for (auto&& [m, k, nnz] : util::dims) {
    for (auto&& n : {1, 8, 32, 64, 512}) {
      auto [values, rowptr, colind, shape, _] =
          spblas::generate_csr<T, I>(m, k, nnz);

      spblas::csr_view<T, I> a(values, rowptr, colind, shape, nnz);

      auto [b_values, b_shape] = spblas::generate_dense<T>(k, n);

      std::vector<T> c_values(m * n, 0);

      md::mdspan b(b_values.data(), k, n);
      md::mdspan c(c_values.data(), m, n);

      auto b_view = spblas::scaled(scaling_factor, b);

      spblas::multiply(a, b_view, c);

      std::vector<T> c_ref(m * n, 0);

      for (I i = 0; i < m; i++) {
        for (I k_ptr = rowptr[i]; k_ptr < rowptr[i + 1]; k_ptr++) {
          I k = colind[k_ptr];
          T v = values[k_ptr];

          for (I j = 0; j < n; j++) {
            c_ref[i * n + j] += v * scaling_factor * b_values[k * n + j];
          }
        }
      }

      for (I i = 0; i < c_ref.size(); i++) {
        EXPECT_EQ_(c_ref[i], c_values[i]);
      }
    }
  }
}

TEST(CsrView, SpMM_Aopt) {
  namespace md = spblas::__mdspan;

  using T = float;
  using I = spblas::index_t;

  for (auto [m, k, nnz] : util::dims) {
    for (auto n : {1, 8, 32, 64, 512}) {
      auto [values, rowptr, colind, shape, _] =
          spblas::generate_csr<T, I>(m, k, nnz);

      spblas::csr_view<T, I> a(values, rowptr, colind, shape, nnz);
      spblas::matrix_opt a_opt(a);

      auto [b_values, b_shape] = spblas::generate_dense<T>(k, n);

      std::vector<T> c_values(m * n, 0);

      md::mdspan b(b_values.data(), k, n);
      md::mdspan c(c_values.data(), m, n);

      spblas::multiply(a_opt, b, c);

      std::vector<T> c_ref(m * n, 0);

      for (I i = 0; i < m; i++) {
        for (I k_ptr = rowptr[i]; k_ptr < rowptr[i + 1]; k_ptr++) {
          I k = colind[k_ptr];
          T v = values[k_ptr];

          for (I j = 0; j < n; j++) {
            c_ref[i * n + j] += v * b_values[k * n + j];
          }
        }
      }

      for (I i = 0; i < c_ref.size(); i++) {
        EXPECT_EQ_(c_ref[i], c_values[i]);
      }
    }
  }
}

TEST(CscView, SpGEMM) {
  using T = float;
  using I = spblas::index_t;

  for (auto&& [m, k, nnz] : util::dims) {
    for (auto&& n : {m, k}) {
      auto [a_values, a_colptr, a_rowind, a_shape, a_nnz] =
          spblas::generate_csc<T, I>(m, k, nnz);

      auto [b_values, b_colptr, b_rowind, b_shape, b_nnz] =
          spblas::generate_csc<T, I>(k, n, nnz);

      spblas::csc_view<T, I> a(a_values, a_colptr, a_rowind, a_shape, a_nnz);
      spblas::csc_view<T, I> b(b_values, b_colptr, b_rowind, b_shape, b_nnz);

      std::vector<I> c_colptr(n + 1);

      spblas::csc_view<T, I> c(nullptr, c_colptr.data(), nullptr, {m, n}, 0);

      auto info = spblas::multiply_compute(a, b, c);

      std::vector<T> c_values(info.result_nnz());
      std::vector<I> c_rowind(info.result_nnz());

      c.update(c_values, c_colptr, c_rowind);

      spblas::multiply_fill(info, a, b, c);

      spblas::__backend::spa_accumulator<T, I> c_column_ref(
          spblas::__backend::shape(c)[0]);

      spblas::__backend::spa_accumulator<T, I> c_column_acc(
          spblas::__backend::shape(c)[0]);

      for (auto&& [j, b_column] : spblas::__backend::columns(b)) {
        c_column_ref.clear();
        for (auto&& [k, b_v] : b_column) {
          auto&& a_column = spblas::__backend::lookup_column(a, k);

          for (auto&& [i, a_v] : a_column) {
            c_column_ref[i] += a_v * b_v;
          }
        }

        auto&& c_column = spblas::__backend::lookup_column(c, j);

        // Accumulate output into `c_column_acc` so that we can allow
        // duplicate column indices.
        c_column_acc.clear();
        for (auto&& [i, c_v] : c_column) {
          c_column_acc[i] += c_v;
        }

        for (auto&& [i, c_v] : c_column) {
          EXPECT_EQ_(c_column_ref[i], c_column_acc[i]);
        }

        EXPECT_EQ(c_column_ref.size(), c_column_acc.size());
      }
    }
  }
}
