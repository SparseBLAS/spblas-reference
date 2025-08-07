
#include "../util.hpp"
#include <spblas/spblas.hpp>

#include <gtest/gtest.h>
#include <thrust/device_vector.h>

using value_t = float;
using index_t = spblas::index_t;
using offset_t = spblas::offset_t;

TEST(thrust_CsrView, SpGEMM) {
  for (auto&& [m, k, nnz] : util::dims) {
    for (auto&& n : {m, k}) {
      auto [a_values, a_rowptr, a_colind, a_shape, a_nnz] =
          spblas::generate_csr<value_t, index_t, offset_t>(m, k, nnz);
      thrust::device_vector<value_t> d_a_values(a_values);
      thrust::device_vector<offset_t> d_a_rowptr(a_rowptr);
      thrust::device_vector<index_t> d_a_colind(a_colind);
      spblas::csr_view<value_t, index_t, offset_t> d_a(
          d_a_values.data().get(), d_a_rowptr.data().get(),
          d_a_colind.data().get(), a_shape, a_nnz);
      spblas::csr_view<value_t, index_t, offset_t> a(a_values, a_rowptr,
                                                     a_colind, a_shape, a_nnz);

      auto [b_values, b_rowptr, b_colind, b_shape, b_nnz] =
          spblas::generate_csr<value_t, index_t, offset_t>(k, n, nnz);
      thrust::device_vector<value_t> d_b_values(b_values);
      thrust::device_vector<offset_t> d_b_rowptr(b_rowptr);
      thrust::device_vector<index_t> d_b_colind(b_colind);
      spblas::csr_view<value_t, index_t, offset_t> d_b(
          d_b_values.data().get(), d_b_rowptr.data().get(),
          d_b_colind.data().get(), b_shape, b_nnz);
      spblas::csr_view<value_t, index_t, offset_t> b(b_values, b_rowptr,
                                                     b_colind, b_shape, b_nnz);

      thrust::device_vector<offset_t> d_c_rowptr(m + 1);

      spblas::csr_view<value_t, index_t, offset_t> d_c(
          nullptr, d_c_rowptr.data().get(), nullptr, {m, n}, 0);

      spblas::spgemm_state_t state;
      spblas::multiply_compute(state, d_a, d_b, d_c);
      auto nnz = state.result_nnz();
      thrust::device_vector<value_t> d_c_values(nnz);
      thrust::device_vector<index_t> d_c_colind(nnz);
      std::span<value_t> d_c_values_span(d_c_values.data().get(), nnz);
      std::span<offset_t> d_c_rowptr_span(d_c_rowptr.data().get(), m + 1);
      std::span<index_t> d_c_colind_span(d_c_colind.data().get(), nnz);
      d_c.update(d_c_values_span, d_c_rowptr_span, d_c_colind_span, {m, n},
                 nnz);

      spblas::multiply_fill(state, d_a, d_b, d_c);

      std::vector<value_t> c_values(nnz);
      std::vector<offset_t> c_rowptr(m + 1);
      std::vector<index_t> c_colind(nnz);
      thrust::copy(d_c_values.begin(), d_c_values.end(), c_values.begin());
      thrust::copy(d_c_rowptr.begin(), d_c_rowptr.end(), c_rowptr.begin());
      thrust::copy(d_c_colind.begin(), d_c_colind.end(), c_colind.begin());
      spblas::csr_view<value_t, index_t, offset_t> c(c_values, c_rowptr,
                                                     c_colind, {m, n}, nnz);

      spblas::__backend::spa_accumulator<value_t, index_t> c_row_ref(
          spblas::__backend::shape(c)[1]);

      spblas::__backend::spa_accumulator<value_t, index_t> c_row_acc(
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

TEST(thrust_CsrView, SpGEMM_AScaled) {
  value_t alpha = 2.0f;
  for (auto&& [m, k, nnz] : util::dims) {
    for (auto&& n : {m, k}) {
      auto [a_values, a_rowptr, a_colind, a_shape, a_nnz] =
          spblas::generate_csr<value_t, index_t, offset_t>(m, k, nnz);
      thrust::device_vector<value_t> d_a_values(a_values);
      thrust::device_vector<offset_t> d_a_rowptr(a_rowptr);
      thrust::device_vector<index_t> d_a_colind(a_colind);
      spblas::csr_view<value_t, index_t, offset_t> d_a(
          d_a_values.data().get(), d_a_rowptr.data().get(),
          d_a_colind.data().get(), a_shape, a_nnz);
      spblas::csr_view<value_t, index_t, offset_t> a(a_values, a_rowptr,
                                                     a_colind, a_shape, a_nnz);

      auto [b_values, b_rowptr, b_colind, b_shape, b_nnz] =
          spblas::generate_csr<value_t, index_t, offset_t>(k, n, nnz);
      thrust::device_vector<value_t> d_b_values(b_values);
      thrust::device_vector<offset_t> d_b_rowptr(b_rowptr);
      thrust::device_vector<index_t> d_b_colind(b_colind);
      spblas::csr_view<value_t, index_t, offset_t> d_b(
          d_b_values.data().get(), d_b_rowptr.data().get(),
          d_b_colind.data().get(), b_shape, b_nnz);
      spblas::csr_view<value_t, index_t, offset_t> b(b_values, b_rowptr,
                                                     b_colind, b_shape, b_nnz);

      thrust::device_vector<offset_t> d_c_rowptr(m + 1);

      spblas::csr_view<value_t, index_t, offset_t> d_c(
          nullptr, d_c_rowptr.data().get(), nullptr, {m, n}, 0);

      spblas::spgemm_state_t state;
      spblas::multiply_compute(state, spblas::scaled(alpha, d_a), d_b, d_c);
      auto nnz = state.result_nnz();
      thrust::device_vector<value_t> d_c_values(nnz);
      thrust::device_vector<index_t> d_c_colind(nnz);
      std::span<value_t> d_c_values_span(d_c_values.data().get(), nnz);
      std::span<offset_t> d_c_rowptr_span(d_c_rowptr.data().get(), m + 1);
      std::span<index_t> d_c_colind_span(d_c_colind.data().get(), nnz);
      d_c.update(d_c_values_span, d_c_rowptr_span, d_c_colind_span, {m, n},
                 nnz);

      spblas::multiply_fill(state, spblas::scaled(alpha, d_a), d_b, d_c);

      std::vector<value_t> c_values(nnz);
      std::vector<offset_t> c_rowptr(m + 1);
      std::vector<index_t> c_colind(nnz);
      thrust::copy(d_c_values.begin(), d_c_values.end(), c_values.begin());
      thrust::copy(d_c_rowptr.begin(), d_c_rowptr.end(), c_rowptr.begin());
      thrust::copy(d_c_colind.begin(), d_c_colind.end(), c_colind.begin());
      spblas::csr_view<value_t, index_t, offset_t> c(c_values, c_rowptr,
                                                     c_colind, {m, n}, nnz);

      spblas::__backend::spa_accumulator<value_t, index_t> c_row_ref(
          spblas::__backend::shape(c)[1]);

      spblas::__backend::spa_accumulator<value_t, index_t> c_row_acc(
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

TEST(thrust_CsrView, SpGEMM_BScaled) {
  value_t alpha = 2.0f;
  for (auto&& [m, k, nnz] : util::dims) {
    for (auto&& n : {m, k}) {
      auto [a_values, a_rowptr, a_colind, a_shape, a_nnz] =
          spblas::generate_csr<value_t, index_t, offset_t>(m, k, nnz);
      thrust::device_vector<value_t> d_a_values(a_values);
      thrust::device_vector<offset_t> d_a_rowptr(a_rowptr);
      thrust::device_vector<index_t> d_a_colind(a_colind);
      spblas::csr_view<value_t, index_t, offset_t> d_a(
          d_a_values.data().get(), d_a_rowptr.data().get(),
          d_a_colind.data().get(), a_shape, a_nnz);
      spblas::csr_view<value_t, index_t, offset_t> a(a_values, a_rowptr,
                                                     a_colind, a_shape, a_nnz);

      auto [b_values, b_rowptr, b_colind, b_shape, b_nnz] =
          spblas::generate_csr<value_t, index_t, offset_t>(k, n, nnz);
      thrust::device_vector<value_t> d_b_values(b_values);
      thrust::device_vector<offset_t> d_b_rowptr(b_rowptr);
      thrust::device_vector<index_t> d_b_colind(b_colind);
      spblas::csr_view<value_t, index_t, offset_t> d_b(
          d_b_values.data().get(), d_b_rowptr.data().get(),
          d_b_colind.data().get(), b_shape, b_nnz);
      spblas::csr_view<value_t, index_t, offset_t> b(b_values, b_rowptr,
                                                     b_colind, b_shape, b_nnz);

      thrust::device_vector<offset_t> d_c_rowptr(m + 1);

      spblas::csr_view<value_t, index_t, offset_t> d_c(
          nullptr, d_c_rowptr.data().get(), nullptr, {m, n}, 0);

      spblas::spgemm_state_t state;
      spblas::multiply_compute(state, d_a, spblas::scaled(alpha, d_b), d_c);
      auto nnz = state.result_nnz();
      thrust::device_vector<value_t> d_c_values(nnz);
      thrust::device_vector<index_t> d_c_colind(nnz);
      std::span<value_t> d_c_values_span(d_c_values.data().get(), nnz);
      std::span<offset_t> d_c_rowptr_span(d_c_rowptr.data().get(), m + 1);
      std::span<index_t> d_c_colind_span(d_c_colind.data().get(), nnz);
      d_c.update(d_c_values_span, d_c_rowptr_span, d_c_colind_span, {m, n},
                 nnz);

      spblas::multiply_fill(state, d_a, spblas::scaled(alpha, d_b), d_c);

      std::vector<value_t> c_values(nnz);
      std::vector<offset_t> c_rowptr(m + 1);
      std::vector<index_t> c_colind(nnz);
      thrust::copy(d_c_values.begin(), d_c_values.end(), c_values.begin());
      thrust::copy(d_c_rowptr.begin(), d_c_rowptr.end(), c_rowptr.begin());
      thrust::copy(d_c_colind.begin(), d_c_colind.end(), c_colind.begin());
      spblas::csr_view<value_t, index_t, offset_t> c(c_values, c_rowptr,
                                                     c_colind, {m, n}, nnz);

      spblas::__backend::spa_accumulator<value_t, index_t> c_row_ref(
          spblas::__backend::shape(c)[1]);

      spblas::__backend::spa_accumulator<value_t, index_t> c_row_acc(
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
