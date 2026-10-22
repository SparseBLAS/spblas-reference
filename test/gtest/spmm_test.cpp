#include <gtest/gtest.h>

#include <complex>
#include <cstdint>

#include "util.hpp"
#include <spblas/spblas.hpp>

namespace {

// Y = beta * Z (or 0), then Y += alpha * (A * X); X, Y, Z row-major.
template <typename T, typename I, typename S>
std::vector<T>
spmm_ref(const std::vector<T>& values, const std::vector<I>& rowptr,
         const std::vector<I>& colind, std::size_t m, std::size_t n,
         const std::vector<T>& x, S alpha, S beta, const std::vector<T>& z) {
  std::vector<T> y(m * n);
  for (std::size_t i = 0; i < m; i++) {
    for (std::size_t j = 0; j < n; j++) {
      y[i * n + j] = beta == S(0) ? T(0) : T(beta) * z[i * n + j];
    }
    for (auto p = rowptr[i]; p < rowptr[i + 1]; p++) {
      for (std::size_t j = 0; j < n; j++) {
        y[i * n + j] += T(alpha) * (values[p] * x[colind[p] * n + j]);
      }
    }
  }
  return y;
}

} // namespace

TEST(Multiply, SpMM_Csr) {
  using T = float;
  using I = spblas::index_t;

  for (auto&& [m, k, nnz] : util::dims) {
    for (auto&& n : {1, 8, 40}) {
      auto [values, rowptr, colind, shape, _] =
          spblas::generate_csr<T, I>(m, k, nnz);
      spblas::csr_wrapper<T, I> a(values, rowptr, colind, shape, nnz);

      T alpha = -0.5;
      auto x_values = util::make_values<T>(k * n);
      std::vector<T> y_values(m * n, T(5));
      spblas::mdspan_row_major<T, I> x(x_values.data(), k, n);
      spblas::mdspan_row_major<T, I> y(y_values.data(), m, n);

      spblas::operation_state state;
      spblas::multiply_inspect(state, alpha, a, x, y);
      spblas::multiply(state, alpha, a, x, y);

      auto y_ref = spmm_ref(values, rowptr, colind, m, n, x_values, alpha, T(0),
                            y_values);
      for (std::size_t i = 0; i < y_ref.size(); i++) {
        EXPECT_EQ_(y_ref[i], y_values[i]);
      }
    }
  }
}

TEST(Multiply, SpMM_BetaZ) {
  using T = double;
  using I = spblas::index_t;

  for (auto&& [m, k, nnz] : util::dims) {
    for (T beta : {T(0), T(2)}) {
      std::size_t n = 8;
      auto [values, rowptr, colind, shape, _] =
          spblas::generate_csr<T, I>(m, k, nnz);
      spblas::csr_wrapper<T, I> a(values, rowptr, colind, shape, nnz);

      T alpha = 1.5;
      auto x_values = util::make_values<T>(k * n);
      auto z_values = util::make_values<T>(m * n, 2);
      std::vector<T> y_values(m * n);
      spblas::mdspan_row_major<T, I> x(x_values.data(), k, n);
      spblas::mdspan_row_major<T, I> z(z_values.data(), m, n);
      spblas::mdspan_row_major<T, I> y(y_values.data(), m, n);

      auto y_ref = spmm_ref(values, rowptr, colind, m, n, x_values, alpha, beta,
                            z_values);

      spblas::operation_state state;
      spblas::multiply(util::test_policy{}, state, alpha, a, x, beta, z, y);
      for (std::size_t i = 0; i < y_ref.size(); i++) {
        EXPECT_EQ_(y_ref[i], y_values[i]);
      }

      // Z aliasing Y.
      spblas::multiply(state, alpha, a, x, beta, z, z);
      for (std::size_t i = 0; i < y_ref.size(); i++) {
        EXPECT_EQ_(y_ref[i], z_values[i]);
      }
    }
  }
}

TEST(Multiply, SpMM_ColumnMajor) {
  using T = float;
  using I = spblas::index_t;

  for (auto&& [m, k, nnz] : util::dims) {
    std::size_t n = 8;
    auto [values, rowptr, colind, shape, _] =
        spblas::generate_csr<T, I>(m, k, nnz);
    spblas::csr_wrapper<T, I> a(values, rowptr, colind, shape, nnz);

    auto x_row = util::make_values<T>(k * n);
    std::vector<T> x_col(k * n);
    for (std::size_t i = 0; i < k; i++) {
      for (std::size_t j = 0; j < n; j++) {
        x_col[j * k + i] = x_row[i * n + j];
      }
    }
    std::vector<T> y_col(m * n);
    spblas::mdspan_col_major<T, I> x(x_col.data(), k, n);
    spblas::mdspan_col_major<T, I> y(y_col.data(), m, n);

    spblas::operation_state state;
    spblas::multiply(state, T(2), a, x, y);

    auto y_ref =
        spmm_ref(values, rowptr, colind, m, n, x_row, T(2), T(0), y_col);
    for (std::size_t i = 0; i < m; i++) {
      for (std::size_t j = 0; j < n; j++) {
        EXPECT_EQ_(y_ref[i * n + j], y_col[j * m + i]);
      }
    }
  }
}

TEST(Multiply, SpMM_CscAndMatrixOpt) {
  using T = float;
  using I = spblas::index_t;

  for (auto&& [m, k, nnz] : util::dims) {
    std::size_t n = 8;
    auto [values, rowptr, colind, shape, _] =
        spblas::generate_csr<T, I>(m, k, nnz);
    spblas::csr_wrapper<T, I> a(values, rowptr, colind, shape, nnz);

    // A in CSC, built with the transpose operation.
    std::vector<I> t_rowptr(k + 1);
    std::vector<I> t_colind(nnz);
    std::vector<T> t_values(nnz);
    spblas::csr_wrapper<T, I> a_t(t_values, t_rowptr, t_colind, {I(k), I(m)},
                                  nnz);
    spblas::operation_state t_state;
    spblas::transpose(t_state, a, a_t);
    spblas::csc_wrapper<T, I> a_csc(t_values, t_rowptr, t_colind, {I(m), I(k)},
                                    nnz);

    auto x_values = util::make_values<T>(k * n);
    spblas::mdspan_row_major<T, I> x(x_values.data(), k, n);
    std::vector<T> y_ref(m * n), y_csc(m * n), y_opt(m * n);

    spblas::operation_state state;
    spblas::multiply(state, T(3), a, x,
                     spblas::mdspan_row_major<T, I>(y_ref.data(), m, n));
    spblas::multiply(state, T(3), a_csc, x,
                     spblas::mdspan_row_major<T, I>(y_csc.data(), m, n));
    spblas::matrix_opt a_opt(a);
    spblas::multiply(state, T(3), a_opt, x,
                     spblas::mdspan_row_major<T, I>(y_opt.data(), m, n));

    for (std::size_t i = 0; i < y_ref.size(); i++) {
      EXPECT_EQ_(y_ref[i], y_csc[i]);
      EXPECT_EQ_(y_ref[i], y_opt[i]);
    }
  }
}

TEST(Multiply, SpMM_Conjugated) {
  using T = std::complex<double>;
  using I = spblas::index_t;

  for (auto&& [m, k, nnz] : util::dims) {
    std::size_t n = 4;
    auto [values, rowptr, colind, shape, _] =
        spblas::generate_csr<T, I>(m, k, nnz);
    spblas::csr_wrapper<T, I> a(values, rowptr, colind, shape, nnz);

    std::vector<T> x_values(k * n, T(1, 2));
    std::vector<T> y_values(m * n);
    spblas::mdspan_row_major<T, I> x(x_values.data(), k, n);
    spblas::mdspan_row_major<T, I> y(y_values.data(), m, n);

    spblas::operation_state state;
    spblas::multiply(state, T(1), spblas::view::conjugate(a), x, y);

    std::vector<T> conj_values(values.size());
    std::transform(values.begin(), values.end(), conj_values.begin(),
                   [](T v) { return std::conj(v); });
    auto y_ref = spmm_ref(conj_values, rowptr, colind, m, n, x_values, T(1),
                          T(0), y_values);
    for (std::size_t i = 0; i < y_ref.size(); i++) {
      EXPECT_EQ_(y_ref[i].real(), y_values[i].real());
      EXPECT_EQ_(y_ref[i].imag(), y_values[i].imag());
    }
  }
}

TEST(Multiply, SpMM_DimensionMismatchThrows) {
  using T = float;
  using I = spblas::index_t;

  auto [values, rowptr, colind, shape, nnz] =
      spblas::generate_csr<T, I>(10, 20, 30);
  spblas::csr_wrapper<T, I> a(values, rowptr, colind, shape, nnz);

  std::vector<T> x_values(20 * 4), y_values(10 * 4), z_values(10 * 3);
  spblas::mdspan_row_major<T, I> x(x_values.data(), 20, 4);
  spblas::mdspan_row_major<T, I> y(y_values.data(), 10, 4);
  spblas::mdspan_row_major<T, I> z(z_values.data(), 10, 3);
  spblas::mdspan_row_major<T, I> x_bad(x_values.data(), 19, 4);

  spblas::operation_state state;
  EXPECT_THROW(spblas::multiply(state, T(1), a, x_bad, y),
               std::invalid_argument);
  EXPECT_THROW(spblas::multiply(state, T(1), a, x, T(1), z, y),
               std::invalid_argument);
}

TEST(Multiply, SpMM_TransposedDense) {
  using T = double;
  using I = spblas::index_t;

  for (auto&& [m, k, nnz] : util::dims) {
    std::size_t n = 5;
    auto [values, rowptr, colind, shape, _] =
        spblas::generate_csr<T, I>(m, k, nnz);
    spblas::csr_wrapper<T, I> a(values, rowptr, colind, shape, nnz);

    // Y = A * X^T with X stored as n x k.
    auto x_values = util::make_values<T>(n * k);
    std::vector<T> y_values(m * n);
    spblas::mdspan_row_major<T, I> x(x_values.data(), n, k);
    spblas::mdspan_row_major<T, I> y(y_values.data(), m, n);

    spblas::operation_state state;
    spblas::multiply(state, T(1), a, spblas::view::transpose(x), y);

    std::vector<T> y_ref(m * n, T(0));
    for (std::size_t i = 0; i < m; i++) {
      for (auto p = rowptr[i]; p < rowptr[i + 1]; p++) {
        for (std::size_t j = 0; j < n; j++) {
          y_ref[i * n + j] += values[p] * x_values[j * k + colind[p]];
        }
      }
    }
    for (std::size_t i = 0; i < y_ref.size(); i++) {
      EXPECT_EQ_(y_ref[i], y_values[i]);
    }
  }
}

TEST(Multiply, SpMM_Int64IndexMatrixOptInspect) {
  using T = float;
  using I = std::int64_t;

  for (auto&& [m, k, nnz] : util::dims) {
    std::size_t n = 6;
    auto [values, rowptr, colind, shape, _] =
        spblas::generate_csr<T, I>(m, k, nnz);
    spblas::csr_wrapper<T, I> a(values, rowptr, colind, shape, nnz);
    spblas::matrix_opt a_opt(a);

    auto x_values = util::make_values<T>(k * n);
    std::vector<T> y_values(m * n);
    spblas::mdspan_row_major<T, I> x(x_values.data(), k, n);
    spblas::mdspan_row_major<T, I> y(y_values.data(), m, n);

    spblas::operation_state state;
    spblas::multiply_inspect(state, T(1.5), a_opt, x, y);
    spblas::multiply(state, T(1.5), a_opt, x, y);

    auto y_ref = spmm_ref(values, rowptr, colind, m, n, x_values, T(1.5), T(0),
                          y_values);
    for (std::size_t i = 0; i < y_ref.size(); i++) {
      EXPECT_EQ_(y_ref[i], y_values[i]);
    }
  }
}
