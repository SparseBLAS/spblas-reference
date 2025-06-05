
#include "../util.hpp"
#include <spblas/spblas.hpp>

#include <gtest/gtest.h>
#include <thrust/device_vector.h>

using value_t = float;
using index_t = spblas::index_t;
using offset_t = spblas::offset_t;

TEST(CsrView, SpGEMMReuse) {
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

      std::vector<offset_t> c_rowptr(m + 1);
      thrust::device_vector<offset_t> d_c_rowptr(c_rowptr);

      spblas::csr_view<value_t, index_t, offset_t> d_c(
          nullptr, d_c_rowptr.data().get(), nullptr, {m, n}, 0);

      spblas::spgemm_state_t state;
      spblas::multiply_symbolic_compute(state, d_a, d_b, d_c);
      auto nnz = state.result_nnz();
      std::vector<value_t> c_values(nnz);
      std::vector<index_t> c_colind(nnz);
      thrust::device_vector<value_t> d_c_values(c_values);
      thrust::device_vector<index_t> d_c_colind(c_colind);
      std::span<value_t> d_c_values_span(d_c_values.data().get(), nnz);
      std::span<offset_t> d_c_rowptr_span(d_c_rowptr.data().get(), m + 1);
      std::span<index_t> d_c_colind_span(d_c_colind.data().get(), nnz);
      d_c.update(d_c_values_span, d_c_rowptr_span, d_c_colind_span, {m, n},
                 nnz);

      spblas::multiply_symbolic_fill(state, d_a, d_b, d_c);
      std::mt19937 g(0);
      for (int i = 0; i < 3; i++) {
        // we can change the value of a and b but only need to call
        // multiply_numeric answer here.
        if (i != 0) {
          // regenerate value of a and b;
          std::uniform_real_distribution val_dist(0.0, 100.0);
          for (auto& v : a_values) {
            v = val_dist(g);
          }
          for (auto& v : b_values) {
            v = val_dist(g);
          }
          thrust::copy(a_values.begin(), a_values.end(), d_a_values.begin());
          thrust::copy(b_values.begin(), b_values.end(), d_b_values.begin());
        }
        spblas::multiply_numeric(state, d_a, d_b, d_c);
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
}

TEST(CsrView, SpGEMMReuse_AScaled) {
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
      std::vector<offset_t> c_rowptr(m + 1);
      thrust::device_vector<offset_t> d_c_rowptr(c_rowptr);

      spblas::csr_view<value_t, index_t, offset_t> d_c(
          nullptr, d_c_rowptr.data().get(), nullptr, {m, n}, 0);

      spblas::spgemm_state_t state;
      spblas::multiply_symbolic_compute(state, scaled(alpha, d_a), d_b, d_c);
      auto nnz = state.result_nnz();
      std::vector<value_t> c_values(nnz);
      std::vector<index_t> c_colind(nnz);
      thrust::device_vector<value_t> d_c_values(c_values);
      thrust::device_vector<index_t> d_c_colind(c_colind);
      std::span<value_t> d_c_values_span(d_c_values.data().get(), nnz);
      std::span<offset_t> d_c_rowptr_span(d_c_rowptr.data().get(), m + 1);
      std::span<index_t> d_c_colind_span(d_c_colind.data().get(), nnz);
      d_c.update(d_c_values_span, d_c_rowptr_span, d_c_colind_span, {m, n},
                 nnz);

      spblas::multiply_symbolic_fill(state, scaled(alpha, d_a), d_b, d_c);
      std::mt19937 g(0);
      for (int i = 0; i < 3; i++) {
        // we can change the value of a and b but only need to call
        // multiply_numeric answer here.
        if (i != 0) {
          // regenerate value of a and b;
          std::uniform_real_distribution val_dist(0.0, 100.0);
          for (auto& v : a_values) {
            v = val_dist(g);
          }
          for (auto& v : b_values) {
            v = val_dist(g);
          }
          thrust::copy(a_values.begin(), a_values.end(), d_a_values.begin());
          thrust::copy(b_values.begin(), b_values.end(), d_b_values.begin());
        }
        spblas::multiply_numeric(state, scaled(alpha, d_a), d_b, d_c);
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
}

TEST(CsrView, SpGEMMReuse_BScaled) {
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
      std::vector<offset_t> c_rowptr(m + 1);
      thrust::device_vector<offset_t> d_c_rowptr(c_rowptr);

      spblas::csr_view<value_t, index_t, offset_t> d_c(
          nullptr, d_c_rowptr.data().get(), nullptr, {m, n}, 0);

      spblas::spgemm_state_t state;
      spblas::multiply_symbolic_compute(state, d_a, scaled(alpha, d_b), d_c);
      auto nnz = state.result_nnz();
      std::vector<value_t> c_values(nnz);
      std::vector<index_t> c_colind(nnz);
      thrust::device_vector<value_t> d_c_values(c_values);
      thrust::device_vector<index_t> d_c_colind(c_colind);
      std::span<value_t> d_c_values_span(d_c_values.data().get(), nnz);
      std::span<offset_t> d_c_rowptr_span(d_c_rowptr.data().get(), m + 1);
      std::span<index_t> d_c_colind_span(d_c_colind.data().get(), nnz);
      d_c.update(d_c_values_span, d_c_rowptr_span, d_c_colind_span, {m, n},
                 nnz);

      spblas::multiply_symbolic_fill(state, d_a, scaled(alpha, d_b), d_c);
      std::mt19937 g(0);
      for (int i = 0; i < 3; i++) {
        // we can change the value of a and b but only need to call
        // multiply_numeric answer here.
        if (i != 0) {
          // regenerate value of a and b;
          std::uniform_real_distribution val_dist(0.0, 100.0);
          for (auto& v : a_values) {
            v = val_dist(g);
          }
          for (auto& v : b_values) {
            v = val_dist(g);
          }
          thrust::copy(a_values.begin(), a_values.end(), d_a_values.begin());
          thrust::copy(b_values.begin(), b_values.end(), d_b_values.begin());
        }
        spblas::multiply_numeric(state, d_a, scaled(alpha, d_b), d_c);
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
}

TEST(CsrView, SpGEMMReuseAndChangePointer) {
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

      std::vector<offset_t> c_rowptr(m + 1);
      thrust::device_vector<offset_t> d_c_rowptr(c_rowptr);

      spblas::csr_view<value_t, index_t, offset_t> d_c(
          nullptr, d_c_rowptr.data().get(), nullptr, {m, n}, 0);

      spblas::spgemm_state_t state;
      spblas::multiply_symbolic_compute(state, d_a, d_b, d_c);
      auto nnz = state.result_nnz();
      std::vector<value_t> c_values(nnz);
      std::vector<index_t> c_colind(nnz);
      thrust::device_vector<value_t> d_c_values(c_values);
      thrust::device_vector<index_t> d_c_colind(c_colind);
      std::span<value_t> d_c_values_span(d_c_values.data().get(), nnz);
      std::span<offset_t> d_c_rowptr_span(d_c_rowptr.data().get(), m + 1);
      std::span<index_t> d_c_colind_span(d_c_colind.data().get(), nnz);
      d_c.update(d_c_values_span, d_c_rowptr_span, d_c_colind_span, {m, n},
                 nnz);

      spblas::multiply_symbolic_fill(state, d_a, d_b, d_c);
      // move the sparsity back to host for later copy
      thrust::copy(d_c_rowptr.begin(), d_c_rowptr.end(), c_rowptr.begin());
      thrust::copy(d_c_colind.begin(), d_c_colind.end(), c_colind.begin());
      std::mt19937 g(0);
      for (int i = 0; i < 3; i++) {
        // regenerate value of a and b;
        std::uniform_real_distribution val_dist(0.0, 100.0);
        for (auto& v : a_values) {
          v = val_dist(g);
        }
        for (auto& v : b_values) {
          v = val_dist(g);
        }
        // create different pointers than the symbolic phase, but they still
        // hold the same sparsity.
        // note. cuda without nvcc can only copy from host to device
        thrust::device_vector<value_t> d_a_values_new(a_values);
        thrust::device_vector<index_t> d_a_colind_new(a_colind);
        thrust::device_vector<index_t> d_a_rowptr_new(a_rowptr);
        thrust::device_vector<value_t> d_b_values_new(b_values);
        thrust::device_vector<index_t> d_b_colind_new(b_colind);
        thrust::device_vector<index_t> d_b_rowptr_new(b_rowptr);
        thrust::device_vector<value_t> d_c_values_new(c_values);
        thrust::device_vector<index_t> d_c_colind_new(c_colind);
        thrust::device_vector<index_t> d_c_rowptr_new(c_rowptr);
        spblas::csr_view<value_t, index_t, offset_t> d_a(
            d_a_values_new.data().get(), d_a_rowptr_new.data().get(),
            d_a_colind_new.data().get(), a_shape, a_nnz);
        spblas::csr_view<value_t, index_t, offset_t> d_b(
            d_b_values_new.data().get(), d_b_rowptr_new.data().get(),
            d_b_colind_new.data().get(), b_shape, b_nnz);
        spblas::csr_view<value_t, index_t, offset_t> d_c(
            d_c_values_new.data().get(), d_c_rowptr_new.data().get(),
            d_c_colind_new.data().get(), {m, n}, nnz);
        // call numeric on new data
        spblas::multiply_numeric(state, d_a, d_b, d_c);
        // move c back to host memory
        thrust::copy(d_c_values_new.begin(), d_c_values_new.end(),
                     c_values.begin());
        thrust::copy(d_c_rowptr_new.begin(), d_c_rowptr_new.end(),
                     c_rowptr.begin());
        thrust::copy(d_c_colind_new.begin(), d_c_colind_new.end(),
                     c_colind.begin());

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
}
