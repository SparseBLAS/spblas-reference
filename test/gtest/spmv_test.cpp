#include <gtest/gtest.h>

#include "util.hpp"
#include <spblas/spblas.hpp>

TEST(CsrView, SpMV) {
  using T = float;
  using I = spblas::index_t;

  for (auto&& [m, n, nnz] : util::dims) {
    auto [values, rowptr, colind, shape, _] =
        spblas::generate_csr<T, I>(m, n, nnz);

    spblas::csr_view<T, I> a(values, rowptr, colind, shape, nnz);

    std::vector<T> b(n, 1);
    std::vector<T> c(m, 0);

    spblas::multiply(a, b, c);

    std::vector<T> c_ref(m, 0);

    for (I i = 0; i < m; i++) {
      for (I j_ptr = rowptr[i]; j_ptr < rowptr[i + 1]; j_ptr++) {
        I j = colind[j_ptr];
        T v = values[j_ptr];

        c_ref[i] += v * b[j];
      }
    }

    for (I i = 0; i < c_ref.size(); i++) {
      EXPECT_EQ_(c_ref[i], c[i]);
    }
  }
}

TEST(CsrView, SpMV_AScaled) {
  using T = float;
  using I = spblas::index_t;

  for (auto&& [m, n, nnz] :
       {std::tuple(1000, 100, 100), std::tuple(100, 1000, 10000),
        std::tuple(40, 40, 1000)}) {
    for (auto&& alpha : {-10, 1, 5}) {
      auto [values, rowptr, colind, shape, _] =
          spblas::generate_csr<T, I>(m, n, nnz);

      spblas::csr_view<T, I> a(values, rowptr, colind, shape, nnz);

      std::vector<T> b(n, 1);
      std::vector<T> c(m, 0);

      spblas::multiply(spblas::scaled(alpha, a), b, c);

      std::vector<T> c_ref(m, 0);

      for (I i = 0; i < m; i++) {
        for (I j_ptr = rowptr[i]; j_ptr < rowptr[i + 1]; j_ptr++) {
          I j = colind[j_ptr];
          T v = values[j_ptr];

          c_ref[i] += v * alpha * b[j];
        }
      }

      for (I i = 0; i < c_ref.size(); i++) {
        EXPECT_EQ_(c_ref[i], c[i]);
      }
    }
  }
}

TEST(CsrView, SpMV_BScaled) {
  using T = float;
  using I = spblas::index_t;

  for (auto&& [m, n, nnz] :
       {std::tuple(1000, 100, 100), std::tuple(100, 1000, 10000),
        std::tuple(40, 40, 1000)}) {
    for (auto&& alpha : {-10, 1, 5}) {
      auto [values, rowptr, colind, shape, _] =
          spblas::generate_csr<T, I>(m, n, nnz);

      spblas::csr_view<T, I> a(values, rowptr, colind, shape, nnz);

      std::vector<T> b(n, 1);
      std::vector<T> c(m, 0);

      spblas::multiply(a, spblas::scaled(alpha, b), c);

      std::vector<T> c_ref(m, 0);

      for (I i = 0; i < m; i++) {
        for (I j_ptr = rowptr[i]; j_ptr < rowptr[i + 1]; j_ptr++) {
          I j = colind[j_ptr];
          T v = values[j_ptr];

          c_ref[i] += v * alpha * b[j];
        }
      }

      for (I i = 0; i < c_ref.size(); i++) {
        EXPECT_EQ_(c_ref[i], c[i]);
      }
    }
  }
}
