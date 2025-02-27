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

TEST(CscView, SpMM) {
  namespace md = spblas::__mdspan;

  using T = float;
  using I = spblas::index_t;

  for (auto [m, k, nnz] : util::dims) {
    for (auto n : {1, 8, 32, 64, 512}) {
      auto [values, colptr, rowind, shape, _] =
          spblas::generate_csc<T, I>(m, k, nnz);

      spblas::csc_view<T, I> a(values, colptr, rowind, shape, nnz);

      auto [b_values, b_shape] = spblas::generate_dense<T>(k, n);

      std::vector<T> c_values(m * n, 0);

      md::mdspan b(b_values.data(), k, n);
      md::mdspan c(c_values.data(), m, n);

      spblas::multiply(a, b, c);

      std::vector<T> c_ref(m * n, 0);

      for (I k_ = 0; k_ < k; k_++) {
        for (I i_ptr = colptr[k_]; i_ptr < colptr[k_ + 1]; i_ptr++) {
          I i = rowind[i_ptr];
          T v = values[i_ptr];

          for (I j = 0; j < n; j++) {
            c_ref[i * n + j] += v * b_values[k_ * n + j];
          }
        }
      }

      for (I i = 0; i < c_ref.size(); i++) {
        EXPECT_EQ_(c_ref[i], c_values[i]);
      }
    }
  }
}
