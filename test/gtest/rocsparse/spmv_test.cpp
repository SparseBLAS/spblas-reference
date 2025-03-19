
#include "../util.hpp"
#include <spblas/spblas.hpp>

#include <gtest/gtest.h>
#include <thrust/device_vector.h>

using value_t = float;
using index_t = spblas::index_t;
using offset_t = spblas::offset_t;

TEST(CsrView, SpMV) {
  for (auto&& [num_rows, num_cols, nnz] : util::dims) {
    auto [values, rowptr, colind, shape, _] =
        spblas::generate_csr<value_t, index_t, offset_t>(num_rows, num_cols,
                                                         nnz);

    std::vector<value_t> b(num_cols, 1);
    std::vector<value_t> c(num_rows, 0);

    thrust::device_vector<value_t> d_values(values);
    thrust::device_vector<offset_t> d_rowptr(rowptr);
    thrust::device_vector<index_t> d_colind(colind);

    thrust::device_vector<value_t> d_b(b);
    thrust::device_vector<value_t> d_c(c);

    spblas::csr_view<value_t, index_t, offset_t> a(
        d_values.data().get(), d_rowptr.data().get(), d_colind.data().get(),
        shape, nnz);

    std::span<value_t> b_span(d_b.data().get(), num_cols);
    std::span<value_t> c_span(d_c.data().get(), num_rows);

    spblas::spmv_state_t state;
    spblas::multiply(state, a, b_span, c_span);

    thrust::copy(d_c.begin(), d_c.end(), c.begin());

    std::vector<value_t> c_ref(num_rows, 0);
    for (index_t i = 0; i < num_rows; i++) {
      for (auto j_ptr = rowptr[i]; j_ptr < rowptr[i + 1]; j_ptr++) {
        auto j = colind[j_ptr];
        auto v = values[j_ptr];

        c_ref[i] += v * b[j];
      }
    }

    for (index_t i = 0; i < c_ref.size(); i++) {
      EXPECT_EQ_(c_ref[i], c[i]);
    }
  }
}

TEST(CsrView, SpMV_Ascaled) {
  for (auto&& [num_rows, num_cols, nnz] :
       {std::tuple(1000, 100, 100), std::tuple(100, 1000, 10000),
        std::tuple(40, 40, 1000)}) {
    for (auto&& alpha : {-10, 1, 5}) {
      auto [values, rowptr, colind, shape, _] =
          spblas::generate_csr<value_t, index_t, offset_t>(num_rows, num_cols,
                                                           nnz);

      std::vector<value_t> b(num_cols, 1);
      std::vector<value_t> c(num_rows, 0);

      thrust::device_vector<value_t> d_values(values);
      thrust::device_vector<offset_t> d_rowptr(rowptr);
      thrust::device_vector<index_t> d_colind(colind);

      thrust::device_vector<value_t> d_b(b);
      thrust::device_vector<value_t> d_c(c);

      spblas::csr_view<value_t, index_t, offset_t> a(
          d_values.data().get(), d_rowptr.data().get(), d_colind.data().get(),
          shape, nnz);

      std::span<value_t> b_span(d_b.data().get(), num_cols);
      std::span<value_t> c_span(d_c.data().get(), num_rows);

      spblas::spmv_state_t state;
      spblas::multiply(state, spblas::scaled(alpha, a), b_span, c_span);

      thrust::copy(d_c.begin(), d_c.end(), c.begin());

      std::vector<value_t> c_ref(num_rows, 0);
      for (index_t i = 0; i < num_rows; i++) {
        for (auto j_ptr = rowptr[i]; j_ptr < rowptr[i + 1]; j_ptr++) {
          auto j = colind[j_ptr];
          auto v = values[j_ptr];

          c_ref[i] += alpha * v * b[j];
        }
      }

      for (index_t i = 0; i < c_ref.size(); i++) {
        EXPECT_EQ_(c_ref[i], c[i]);
      }
    }
  }
}

TEST(CsrView, SpMV_BScaled) {
  for (auto&& [num_rows, num_cols, nnz] :
       {std::tuple(1000, 100, 100), std::tuple(100, 1000, 10000),
        std::tuple(40, 40, 1000)}) {
    for (auto&& alpha : {-10, 1, 5}) {
      auto [values, rowptr, colind, shape, _] =
          spblas::generate_csr<value_t, index_t, offset_t>(num_rows, num_cols,
                                                           nnz);

      std::vector<value_t> b(num_cols, 1);
      std::vector<value_t> c(num_rows, 0);

      thrust::device_vector<value_t> d_values(values);
      thrust::device_vector<offset_t> d_rowptr(rowptr);
      thrust::device_vector<index_t> d_colind(colind);

      thrust::device_vector<value_t> d_b(b);
      thrust::device_vector<value_t> d_c(c);

      spblas::csr_view<value_t, index_t, offset_t> a(
          d_values.data().get(), d_rowptr.data().get(), d_colind.data().get(),
          shape, nnz);

      std::span<value_t> b_span(d_b.data().get(), num_cols);
      std::span<value_t> c_span(d_c.data().get(), num_rows);

      spblas::spmv_state_t state;
      spblas::multiply(state, a, spblas::scaled(alpha, b_span), c_span);

      thrust::copy(d_c.begin(), d_c.end(), c.begin());

      std::vector<value_t> c_ref(num_rows, 0);
      for (index_t i = 0; i < num_rows; i++) {
        for (auto j_ptr = rowptr[i]; j_ptr < rowptr[i + 1]; j_ptr++) {
          auto j = colind[j_ptr];
          auto v = values[j_ptr];

          c_ref[i] += v * alpha * b[j];
        }
      }

      for (index_t i = 0; i < c_ref.size(); i++) {
        EXPECT_EQ_(c_ref[i], c[i]);
      }
    }
  }
}
