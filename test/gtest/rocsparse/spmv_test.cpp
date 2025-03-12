#include <gtest/gtest.h>

#include "../util.hpp"
#include "memory.hpp"
#include <spblas/allocator.hpp>
#include <spblas/array.hpp>
#include <spblas/spblas.hpp>

#include <thrust/device_vector.h>

TEST(CsrView, SpMV) {
  using T = float;
  using I = spblas::index_t;
  auto alloc = std::make_shared<default_allocator>();

  for (auto&& [m, n, nnz] : util::dims) {
    auto [values, rowptr, colind, shape, _] =
        spblas::generate_csr<T, I>(m, n, nnz);

    std::vector<T> b(n, 1);
    std::vector<T> c(m, 0);

    thrust::device_vector<T> d_values(values);
    thrust::device_vector<I> d_rowptr(rowptr);
    thrust::device_vector<I> d_colind(colind);

    thrust::device_vector<T> d_b(b);
    thrust::device_vector<T> d_c(c);

    spblas::csr_view<T, I> a(d_values.data().get(), d_rowptr.data().get(),
                             d_colind.data().get(), shape, nnz);

    std::span<T> b_span(d_b.data().get(), n);
    std::span<T> c_span(d_c.data().get(), m);

    spblas::multiply(a, b_span, c_span);

    thrust::copy(d_c.begin(), d_c.end(), c.begin());

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
  auto alloc = std::make_shared<default_allocator>();

  for (auto&& [m, n, nnz] :
       {std::tuple(1000, 100, 100), std::tuple(100, 1000, 10000),
        std::tuple(40, 40, 1000)}) {
    for (auto&& alpha : {-10, 1, 5}) {
      auto [values, rowptr, colind, shape, _] =
          spblas::generate_csr<T, I>(m, n, nnz);

      std::vector<T> b(n, 1);
      std::vector<T> c(m, 0);

      thrust::device_vector<T> d_values(values);
      thrust::device_vector<I> d_rowptr(rowptr);
      thrust::device_vector<I> d_colind(colind);

      thrust::device_vector<T> d_b(b);
      thrust::device_vector<T> d_c(c);

      spblas::csr_view<T, I> a(d_values.data().get(), d_rowptr.data().get(),
                               d_colind.data().get(), shape, nnz);

      std::span<T> b_span(d_b.data().get(), n);
      std::span<T> c_span(d_c.data().get(), m);

      spblas::multiply(spblas::scaled(alpha, a), b_span, c_span);

      thrust::copy(d_c.begin(), d_c.end(), c.begin());

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
  auto alloc = std::make_shared<default_allocator>();

  for (auto&& [m, n, nnz] :
       {std::tuple(1000, 100, 100), std::tuple(100, 1000, 10000),
        std::tuple(40, 40, 1000)}) {
    for (auto&& alpha : {-10, 1, 5}) {
      auto [values, rowptr, colind, shape, _] =
          spblas::generate_csr<T, I>(m, n, nnz);

      std::vector<T> b(n, 1);
      std::vector<T> c(m, 0);

      thrust::device_vector<T> d_values(values);
      thrust::device_vector<I> d_rowptr(rowptr);
      thrust::device_vector<I> d_colind(colind);

      thrust::device_vector<T> d_b(b);
      thrust::device_vector<T> d_c(c);

      spblas::csr_view<T, I> a(d_values.data().get(), d_rowptr.data().get(),
                               d_colind.data().get(), shape, nnz);

      std::span<T> b_span(d_b.data().get(), n);
      std::span<T> c_span(d_c.data().get(), m);

      spblas::multiply(a, spblas::scaled(alpha, b_span), c_span);

      thrust::copy(d_c.begin(), d_c.end(), c.begin());

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
