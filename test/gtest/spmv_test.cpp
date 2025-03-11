#include "memory.hpp"
#include "util.hpp"
#include <gtest/gtest.h>
#include <iostream>
#include <spblas/allocator.hpp>
#include <spblas/array.hpp>
#include <spblas/spblas.hpp>

TEST(CsrView, SpMV) {
  using T = float;
  using I = spblas::index_t;

  for (auto&& [m, n, nnz] : util::dims) {
    auto [values, rowptr, colind, shape, _] =
        spblas::generate_csr<T, I>(m, n, nnz);
    auto dvalues = allocate_device_vector<T>(nnz);
    auto drowptr = allocate_device_vector<I>(m + 1);
    auto dcolind = allocate_device_vector<I>(nnz);
    copy_to_device(nnz, values.data(), dvalues.data());
    copy_to_device(m + 1, rowptr.data(), drowptr.data());
    copy_to_device(nnz, colind.data(), dcolind.data());
    spblas::csr_view<T, I> a(dvalues.data(), drowptr.data(), dcolind.data(),
                             shape, nnz);
    std::vector<T> b(n, 1);
    std::vector<T> c(m, 0);
    auto db = allocate_device_vector<T>(n);
    auto dc = allocate_device_vector<T>(m);
    copy_to_device(n, b.data(), db.data());
    copy_to_device(m, c.data(), dc.data());
    std::span<T> b_span(db.data(), n);
    std::span<T> c_span(dc.data(), m);
    std::vector<T> c_ref(m, 0);
    for (I i = 0; i < m; i++) {
      for (I j_ptr = rowptr[i]; j_ptr < rowptr[i + 1]; j_ptr++) {
        I j = colind[j_ptr];
        T v = values[j_ptr];

        c_ref[i] += v * b[j];
      }
    }

    spblas::multiply(a, b_span, c_span);

    copy_to_host(m, dc.data(), c.data());
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
      auto dvalues = allocate_device_vector<T>(nnz);
      auto drowptr = allocate_device_vector<I>(m + 1);
      auto dcolind = allocate_device_vector<I>(nnz);
      copy_to_device(nnz, values.data(), dvalues.data());
      copy_to_device(m + 1, rowptr.data(), drowptr.data());
      copy_to_device(nnz, colind.data(), dcolind.data());
      spblas::csr_view<T, I> a(dvalues.data(), drowptr.data(), dcolind.data(),
                               shape, nnz);
      std::vector<T> b(n, 1);
      std::vector<T> c(m, 0);
      auto db = allocate_device_vector<T>(n);
      auto dc = allocate_device_vector<T>(m);
      copy_to_device(n, b.data(), db.data());
      copy_to_device(m, c.data(), dc.data());
      std::span<T> b_span(db.data(), n);
      std::span<T> c_span(dc.data(), m);
      std::vector<T> c_ref(m, 0);
      for (I i = 0; i < m; i++) {
        for (I j_ptr = rowptr[i]; j_ptr < rowptr[i + 1]; j_ptr++) {
          I j = colind[j_ptr];
          T v = values[j_ptr];

          c_ref[i] += alpha * v * b[j];
        }
      }

      spblas::multiply(spblas::scaled(alpha, a), b_span, c_span);

      copy_to_host(m, dc.data(), c.data());
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
      auto dvalues = allocate_device_vector<T>(nnz);
      auto drowptr = allocate_device_vector<I>(m + 1);
      auto dcolind = allocate_device_vector<I>(nnz);
      copy_to_device(nnz, values.data(), dvalues.data());
      copy_to_device(m + 1, rowptr.data(), drowptr.data());
      copy_to_device(nnz, colind.data(), dcolind.data());
      spblas::csr_view<T, I> a(dvalues.data(), drowptr.data(), dcolind.data(),
                               shape, nnz);
      std::vector<T> b(n, 1);
      std::vector<T> c(m, 0);
      auto db = allocate_device_vector<T>(n);
      auto dc = allocate_device_vector<T>(m);
      copy_to_device<>(n, b.data(), db.data());
      copy_to_device(m, c.data(), dc.data());
      std::span<T> b_span(db.data(), n);
      std::span<T> c_span(dc.data(), m);
      std::vector<T> c_ref(m, 0);
      for (I i = 0; i < m; i++) {
        for (I j_ptr = rowptr[i]; j_ptr < rowptr[i + 1]; j_ptr++) {
          I j = colind[j_ptr];
          T v = values[j_ptr];

          c_ref[i] += v * alpha * b[j];
        }
      }

      spblas::multiply(a, spblas::scaled(alpha, b_span), c_span);

      copy_to_host(m, dc.data(), c.data());
      for (I i = 0; i < c_ref.size(); i++) {
        EXPECT_EQ_(c_ref[i], c[i]);
      }
    }
  }
}

#ifndef SPBLAS_ENABLE_ROCSPARSE

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

#endif
