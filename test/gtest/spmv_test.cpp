#include <gtest/gtest.h>

#include <complex>

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

TEST(CsrView, SpMV_Ascaled) {
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

          c_ref[i] += alpha * v * b[j];
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

TEST(CscView, SpMV) {
  using T = float;
  using I = spblas::index_t;

  for (auto&& [m, n, nnz] : util::dims) {
    auto [values, colptr, rowind, shape, _] =
        spblas::generate_csc<T, I>(m, n, nnz);

    spblas::csc_view<T, I> a(values, colptr, rowind, shape, nnz);

    std::vector<T> b(n, 1);
    std::vector<T> c(m, 0);

    spblas::multiply(a, b, c);

    std::vector<T> c_ref(m, 0);

    for (I j = 0; j < n; j++) {
      for (I i_ptr = colptr[j]; i_ptr < colptr[j + 1]; i_ptr++) {
        I i = rowind[i_ptr];
        T v = values[i_ptr];

        c_ref[i] += v * b[j];
      }
    }

    for (I i = 0; i < c_ref.size(); i++) {
      EXPECT_EQ_(c_ref[i], c[i]);
    }
  }
}

#ifndef SPBLAS_VENDOR_BACKEND

TEST(CsrView, SpMV_Aconjugated) {
  using T = std::complex<float>;
  using I = spblas::index_t;

  for (auto&& [m, n, nnz] : util::dims) {
    auto [values_real, rowptr, colind, shape, _] =
        spblas::generate_csr<float, I>(m, n, nnz);

    std::vector<T> values(values_real.size());
    for (std::size_t i = 0; i < values.size(); i++) {
      values[i] = T(values_real[i], static_cast<float>((i % 7) + 1));
    }

    spblas::csr_view<T, I> a(values, rowptr, colind, shape, nnz);

    std::vector<T> b(n, T(1.0f, -2.0f));
    std::vector<T> c(m, T(0.0f, 0.0f));

    spblas::multiply(spblas::conjugated(a), b, c);

    std::vector<T> c_ref(m, T(0.0f, 0.0f));

    for (I i = 0; i < m; i++) {
      for (I j_ptr = rowptr[i]; j_ptr < rowptr[i + 1]; j_ptr++) {
        I j = colind[j_ptr];
        T v = values[j_ptr];

        c_ref[i] += std::conj(v) * b[j];
      }
    }

    for (I i = 0; i < c_ref.size(); i++) {
      EXPECT_EQ_(c_ref[i].real(), c[i].real());
      EXPECT_EQ_(c_ref[i].imag(), c[i].imag());
    }
  }
}

TEST(CsrView, SpMV_Bconjugated) {
  using T = std::complex<float>;
  using I = spblas::index_t;

  for (auto&& [m, n, nnz] : util::dims) {
    auto [values_real, rowptr, colind, shape, _] =
        spblas::generate_csr<float, I>(m, n, nnz);

    std::vector<T> values(values_real.size());
    for (std::size_t i = 0; i < values.size(); i++) {
      values[i] = T(values_real[i], static_cast<float>((i % 5) + 1));
    }

    spblas::csr_view<T, I> a(values, rowptr, colind, shape, nnz);

    std::vector<T> b(n, T(1.0f, -2.0f));
    std::vector<T> c(m, T(0.0f, 0.0f));

    spblas::multiply(a, spblas::conjugated(b), c);

    std::vector<T> c_ref(m, T(0.0f, 0.0f));

    for (I i = 0; i < m; i++) {
      for (I j_ptr = rowptr[i]; j_ptr < rowptr[i + 1]; j_ptr++) {
        I j = colind[j_ptr];
        T v = values[j_ptr];

        c_ref[i] += v * std::conj(b[j]);
      }
    }

    for (I i = 0; i < c_ref.size(); i++) {
      EXPECT_EQ_(c_ref[i].real(), c[i].real());
      EXPECT_EQ_(c_ref[i].imag(), c[i].imag());
    }
  }
}

#endif

TEST(CscView, SpMV_Ascaled) {
  using T = float;
  using I = spblas::index_t;

  for (auto&& [m, n, nnz] : util::dims) {
    for (auto&& alpha : {-10, 1, 5}) {
      auto [values, colptr, rowind, shape, _] =
          spblas::generate_csc<T, I>(m, n, nnz);

      spblas::csc_view<T, I> a(values, colptr, rowind, shape, nnz);

      std::vector<T> b(n, 1);
      std::vector<T> c(m, 0);

      spblas::multiply(spblas::scaled(alpha, a), b, c);

      std::vector<T> c_ref(m, 0);

      for (I j = 0; j < n; j++) {
        for (I i_ptr = colptr[j]; i_ptr < colptr[j + 1]; i_ptr++) {
          I i = rowind[i_ptr];
          T v = values[i_ptr];

          c_ref[i] += alpha * v * b[j];
        }
      }

      for (I i = 0; i < c_ref.size(); i++) {
        EXPECT_EQ_(c_ref[i], c[i]);
      }
    }
  }
}

TEST(CscView, SpMV_Bscaled) {
  using T = float;
  using I = spblas::index_t;

  for (auto&& [m, n, nnz] : util::dims) {
    for (auto&& alpha : {-10, 1, 5}) {
      auto [values, colptr, rowind, shape, _] =
          spblas::generate_csc<T, I>(m, n, nnz);

      spblas::csc_view<T, I> a(values, colptr, rowind, shape, nnz);

      std::vector<T> b(n, 1);
      std::vector<T> c(m, 0);

      spblas::multiply(a, spblas::scaled(alpha, b), c);

      std::vector<T> c_ref(m, 0);

      for (I j = 0; j < n; j++) {
        for (I i_ptr = colptr[j]; i_ptr < colptr[j + 1]; i_ptr++) {
          I i = rowind[i_ptr];
          T v = values[i_ptr];

          c_ref[i] += v * alpha * b[j];
        }
      }

      for (I i = 0; i < c_ref.size(); i++) {
        EXPECT_EQ_(c_ref[i], c[i]);
      }
    }
  }
}
