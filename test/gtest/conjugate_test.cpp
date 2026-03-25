#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <limits>
#include <vector>

#include "util.hpp"
#include <spblas/backend/spa_accumulator.hpp>
#include <spblas/spblas.hpp>

namespace {

using T = std::complex<float>;
using I = spblas::index_t;

void expect_complex_eq(const T& expected, const T& actual) {
  auto check = [](float expected_value, float actual_value) {
#ifdef SPBLAS_ENABLE_ONEMKL_SYCL
    constexpr float kEpsilonScale = 1024.0f;
#else
    constexpr float kEpsilonScale = 256.0f;
#endif
    constexpr float kAbsFloor = 1.0e-2f;
    auto epsilon = kEpsilonScale * std::numeric_limits<float>::epsilon();
    auto abs_th = std::numeric_limits<float>::min();
    auto diff = std::abs(expected_value - actual_value);
    auto norm = std::min(std::abs(expected_value) + std::abs(actual_value),
                         std::numeric_limits<float>::max());
    auto abs_error = std::max({abs_th, kAbsFloor, epsilon * norm});
    EXPECT_LE(diff, abs_error);
  };

  check(expected.real(), actual.real());
  check(expected.imag(), actual.imag());
}

} // namespace

#ifndef SPBLAS_ENABLE_ONEMKL_SYCL

TEST(Conjugate, SpMV_MatrixConjugated) {
  for (auto&& [m, n, nnz] : util::dims) {
    auto [values, rowptr, colind, shape, _] =
        spblas::generate_csr<T, I>(m, n, nnz);
    spblas::csr_view<T, I> a(values, rowptr, colind, shape, nnz);

    std::vector<T> b(n, T(1.0f, -2.0f));
    std::vector<T> c(m, T(0.0f, 0.0f));

    spblas::multiply(spblas::conjugated(a), b, c);

    std::vector<T> c_ref(m, T(0.0f, 0.0f));
    for (I i = 0; i < m; i++) {
      for (I j_ptr = rowptr[i]; j_ptr < rowptr[i + 1]; j_ptr++) {
        I j = colind[j_ptr];
        c_ref[i] += std::conj(values[j_ptr]) * b[j];
      }
    }

    for (I i = 0; i < m; i++) {
      expect_complex_eq(c_ref[i], c[i]);
    }
  }
}

TEST(Conjugate, SpMV_VectorConjugated) {
  for (auto&& [m, n, nnz] : util::dims) {
    auto [values, rowptr, colind, shape, _] =
        spblas::generate_csr<T, I>(m, n, nnz);
    spblas::csr_view<T, I> a(values, rowptr, colind, shape, nnz);

    std::vector<T> b(n, T(1.0f, -2.0f));
    std::vector<T> c(m, T(0.0f, 0.0f));

    spblas::multiply(a, spblas::conjugated(b), c);

    std::vector<T> c_ref(m, T(0.0f, 0.0f));
    for (I i = 0; i < m; i++) {
      for (I j_ptr = rowptr[i]; j_ptr < rowptr[i + 1]; j_ptr++) {
        I j = colind[j_ptr];
        c_ref[i] += values[j_ptr] * std::conj(b[j]);
      }
    }

    for (I i = 0; i < m; i++) {
      expect_complex_eq(c_ref[i], c[i]);
    }
  }
}

TEST(Conjugate, SpMM_MatrixConjugated) {
  for (auto&& [m, k, nnz] : util::dims) {
    for (auto n : {1, 8, 32}) {
      auto [values, rowptr, colind, shape, _] =
          spblas::generate_csr<T, I>(m, k, nnz);
      spblas::csr_view<T, I> a(values, rowptr, colind, shape, nnz);

      auto [b_values, b_shape] = spblas::generate_dense<T>(k, n);

      std::vector<T> c_values(m * n, T(0.0f, 0.0f));
      spblas::mdspan_row_major b(b_values.data(), k, n);
      spblas::mdspan_row_major c(c_values.data(), m, n);

      spblas::multiply(spblas::conjugated(a), b, c);

      std::vector<T> c_ref(m * n, T(0.0f, 0.0f));
      for (I i = 0; i < m; i++) {
        for (I k_ptr = rowptr[i]; k_ptr < rowptr[i + 1]; k_ptr++) {
          I kk = colind[k_ptr];
          T v = std::conj(values[k_ptr]);
          for (I j = 0; j < n; j++) {
            c_ref[i * n + j] += v * b_values[kk * n + j];
          }
        }
      }

      for (I i = 0; i < static_cast<I>(c_ref.size()); i++) {
        expect_complex_eq(c_ref[i], c_values[i]);
      }
    }
  }
}

TEST(Conjugate, SpMM_DenseConjugated) {
  for (auto&& [m, k, nnz] : util::dims) {
    for (auto n : {1, 8, 32}) {
      auto [values, rowptr, colind, shape, _] =
          spblas::generate_csr<T, I>(m, k, nnz);
      spblas::csr_view<T, I> a(values, rowptr, colind, shape, nnz);

      auto [b_values, b_shape] = spblas::generate_dense<T>(k, n);

      std::vector<T> c_values(m * n, T(0.0f, 0.0f));
      spblas::mdspan_row_major b(b_values.data(), k, n);
      spblas::mdspan_row_major c(c_values.data(), m, n);

      spblas::multiply(a, spblas::conjugated(b), c);

      std::vector<T> c_ref(m * n, T(0.0f, 0.0f));
      for (I i = 0; i < m; i++) {
        for (I k_ptr = rowptr[i]; k_ptr < rowptr[i + 1]; k_ptr++) {
          I kk = colind[k_ptr];
          T v = values[k_ptr];
          for (I j = 0; j < n; j++) {
            c_ref[i * n + j] += v * std::conj(b_values[kk * n + j]);
          }
        }
      }

      for (I i = 0; i < static_cast<I>(c_ref.size()); i++) {
        expect_complex_eq(c_ref[i], c_values[i]);
      }
    }
  }
}

TEST(Conjugate, SpGEMM_MatrixConjugated) {
  for (auto&& [m, k, nnz] : util::dims) {
    auto [a_values, a_rowptr, a_colind, a_shape, a_nnz] =
        spblas::generate_csr<T, I>(m, k, nnz);
    auto [b_values, b_rowptr, b_colind, b_shape, b_nnz] =
        spblas::generate_csr<T, I>(k, m, nnz);

    spblas::csr_view<T, I> a(a_values, a_rowptr, a_colind, a_shape, a_nnz);
    spblas::csr_view<T, I> b(b_values, b_rowptr, b_colind, b_shape, b_nnz);

    std::vector<I> c_rowptr(m + 1);
    spblas::csr_view<T, I> c(nullptr, c_rowptr.data(), nullptr, {m, m}, 0);

    auto info = spblas::multiply_compute(spblas::conjugated(a), b, c);

    std::vector<T> c_values(info.result_nnz());
    std::vector<I> c_colind(info.result_nnz());
    c.update(c_values, c_rowptr, c_colind);

    spblas::multiply_fill(info, spblas::conjugated(a), b, c);

    spblas::__backend::spa_accumulator<T, I> c_row_ref(m);
    spblas::__backend::spa_accumulator<T, I> c_row_acc(m);

    for (auto&& [i, a_row] : spblas::__backend::rows(a)) {
      c_row_ref.clear();
      for (auto&& [kk, a_v] : a_row) {
        auto&& b_row = spblas::__backend::lookup_row(b, kk);
        for (auto&& [j, b_v] : b_row) {
          c_row_ref[j] += std::conj(a_v) * b_v;
        }
      }

      auto&& c_row = spblas::__backend::lookup_row(c, i);
      c_row_acc.clear();
      for (auto&& [j, c_v] : c_row) {
        c_row_acc[j] += c_v;
      }

      for (auto&& [j, c_v] : c_row) {
        expect_complex_eq(c_row_ref[j], c_row_acc[j]);
      }

      EXPECT_EQ(c_row_ref.size(), c_row_acc.size());
    }
  }
}

#endif

#ifdef SPBLAS_ENABLE_ONEMKL_SYCL

TEST(Conjugate, SpMV_MatrixConjugated) {
  for (auto&& [m, n, nnz] : util::dims) {
    auto [values, colptr, rowind, shape, _] =
        spblas::generate_csc<T, I>(m, n, nnz);
    spblas::csc_view<T, I> a(values, colptr, rowind, shape, nnz);

    std::vector<T> b(n, T(1.0f, -2.0f));
    std::vector<T> c(m, T(0.0f, 0.0f));

    spblas::multiply(spblas::conjugated(a), b, c);

    std::vector<T> c_ref(m, T(0.0f, 0.0f));
    for (I j = 0; j < n; j++) {
      for (I i_ptr = colptr[j]; i_ptr < colptr[j + 1]; i_ptr++) {
        I i = rowind[i_ptr];
        c_ref[i] += std::conj(values[i_ptr]) * b[j];
      }
    }

    for (I i = 0; i < m; i++) {
      expect_complex_eq(c_ref[i], c[i]);
    }
  }
}

TEST(Conjugate, SpMM_MatrixConjugated) {
  namespace md = spblas::__mdspan;

  for (auto&& [m, k, nnz] : util::dims) {
    for (auto n : {1, 8, 32}) {
      auto [values, colptr, rowind, shape, _] =
          spblas::generate_csc<T, I>(m, k, nnz);
      spblas::csc_view<T, I> a(values, colptr, rowind, shape, nnz);

      auto [b_values, b_shape] = spblas::generate_dense<T>(k, n);

      std::vector<T> c_values(m * n, T(0.0f, 0.0f));
      md::mdspan b(b_values.data(), k, n);
      md::mdspan c(c_values.data(), m, n);

      spblas::multiply(spblas::conjugated(a), b, c);

      std::vector<T> c_ref(m * n, T(0.0f, 0.0f));
      for (I col = 0; col < k; col++) {
        for (I i_ptr = colptr[col]; i_ptr < colptr[col + 1]; i_ptr++) {
          I i = rowind[i_ptr];
          T v = std::conj(values[i_ptr]);
          for (I j = 0; j < n; j++) {
            c_ref[i * n + j] += v * b_values[col * n + j];
          }
        }
      }

      for (I i = 0; i < static_cast<I>(c_ref.size()); i++) {
        expect_complex_eq(c_ref[i], c_values[i]);
      }
    }
  }
}

TEST(Conjugate, SpGEMM_MatrixConjugated) {
  for (auto&& [m, k, nnz] : util::dims) {
    auto [a_values, a_colptr, a_rowind, a_shape, a_nnz] =
        spblas::generate_csc<T, I>(m, k, nnz);
    auto [b_values, b_rowptr, b_colind, b_shape, b_nnz] =
        spblas::generate_csr<T, I>(k, m, nnz);

    spblas::csc_view<T, I> a(a_values, a_colptr, a_rowind, a_shape, a_nnz);
    spblas::csr_view<T, I> b(b_values, b_rowptr, b_colind, b_shape, b_nnz);

    std::vector<I> c_rowptr(m + 1);
    spblas::csr_view<T, I> c(nullptr, c_rowptr.data(), nullptr, {m, m}, 0);

    auto info = spblas::multiply_compute(spblas::conjugated(a), b, c);

    std::vector<T> c_values(info.result_nnz());
    std::vector<I> c_colind(info.result_nnz());
    c.update(c_values, c_rowptr, c_colind);

    spblas::multiply_fill(info, spblas::conjugated(a), b, c);

    std::vector<T> c_ref(m * m, T(0.0f, 0.0f));
    for (I col = 0; col < k; col++) {
      for (I i_ptr = a_colptr[col]; i_ptr < a_colptr[col + 1]; i_ptr++) {
        I i = a_rowind[i_ptr];
        T a_v = std::conj(a_values[i_ptr]);
        for (I k_ptr = b_rowptr[col]; k_ptr < b_rowptr[col + 1]; k_ptr++) {
          I j = b_colind[k_ptr];
          c_ref[i * m + j] += a_v * b_values[k_ptr];
        }
      }
    }

    std::vector<T> c_out(m * m, T(0.0f, 0.0f));
    for (I i = 0; i < m; i++) {
      for (I j_ptr = c_rowptr[i]; j_ptr < c_rowptr[i + 1]; j_ptr++) {
        I j = c_colind[j_ptr];
        c_out[i * m + j] += c_values[j_ptr];
      }
    }

    for (I i = 0; i < static_cast<I>(c_ref.size()); i++) {
      expect_complex_eq(c_ref[i], c_out[i]);
    }
  }
}

#endif
