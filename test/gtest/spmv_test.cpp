#include <gtest/gtest.h>

#include <complex>
#include <cstdint>
#include <limits>

#include "util.hpp"
#include <spblas/spblas.hpp>

namespace {

// y = beta * z (or 0), then y += alpha * (A * x), row by row in CSR order;
// the same operation order as the reference implementation.
template <typename T, typename I, typename S>
std::vector<T>
spmv_ref(const std::vector<T>& values, const std::vector<I>& rowptr,
         const std::vector<I>& colind, std::size_t m, const std::vector<T>& x,
         S alpha, S beta, const std::vector<T>& z) {
  std::vector<T> y(m);
  for (std::size_t i = 0; i < m; i++) {
    y[i] = beta == S(0) ? T(0) : T(beta) * z[i];
    for (auto p = rowptr[i]; p < rowptr[i + 1]; p++) {
      y[i] += T(alpha) * (values[p] * x[colind[p]]);
    }
  }
  return y;
}

} // namespace

TEST(Multiply, SpMV_Csr) {
  using T = float;
  using I = spblas::index_t;

  for (auto&& [m, n, nnz] : util::dims) {
    for (T alpha : {T(1), T(-2.5)}) {
      auto [values, rowptr, colind, shape, _] =
          spblas::generate_csr<T, I>(m, n, nnz);
      spblas::csr_wrapper<T, I> a(values, rowptr, colind, shape, nnz);

      auto x = util::make_values<T>(n);
      std::vector<T> y(m, T(7));

      spblas::operation_state state;
      spblas::multiply_inspect(state, alpha, a, x, y);
      spblas::multiply(state, alpha, a, x, y);

      auto y_ref = spmv_ref(values, rowptr, colind, m, x, alpha, T(0), y);
      for (std::size_t i = 0; i < m; i++) {
        EXPECT_EQ_(y_ref[i], y[i]);
      }
    }
  }
}

TEST(Multiply, SpMV_BetaZ) {
  using T = float;
  using I = spblas::index_t;

  for (auto&& [m, n, nnz] : util::dims) {
    for (T beta : {T(0), T(1), T(-3)}) {
      auto [values, rowptr, colind, shape, _] =
          spblas::generate_csr<T, I>(m, n, nnz);
      spblas::csr_wrapper<T, I> a(values, rowptr, colind, shape, nnz);

      T alpha = 2;
      auto x = util::make_values<T>(n);
      auto z = util::make_values<T>(m, 3);
      auto y_ref = spmv_ref(values, rowptr, colind, m, x, alpha, beta, z);

      spblas::operation_state state;

      // z distinct from y
      std::vector<T> y(m, T(99));
      spblas::multiply(state, alpha, a, x, beta, z, y);
      for (std::size_t i = 0; i < m; i++) {
        EXPECT_EQ_(y_ref[i], y[i]);
      }

      // z aliasing y: y = alpha * A * x + beta * y
      std::vector<T> y_alias = z;
      spblas::multiply(state, alpha, a, x, beta, y_alias, y_alias);
      for (std::size_t i = 0; i < m; i++) {
        EXPECT_EQ_(y_ref[i], y_alias[i]);
      }
    }
  }
}

TEST(Multiply, SpMV_BetaZeroDoesNotReadZ) {
  using T = double;
  using I = spblas::index_t;

  auto [m, n, nnz] = util::dims[0];
  auto [values, rowptr, colind, shape, _] =
      spblas::generate_csr<T, I>(m, n, nnz);
  spblas::csr_wrapper<T, I> a(values, rowptr, colind, shape, nnz);

  auto x = util::make_values<T>(n);
  std::vector<T> z(m, std::numeric_limits<T>::quiet_NaN());
  std::vector<T> y(m);

  spblas::operation_state state;
  spblas::multiply(state, T(1), a, x, T(0), z, y);

  for (auto&& v : y) {
    EXPECT_FALSE(std::isnan(v));
  }
}

TEST(Multiply, SpMV_Csc) {
  using T = float;
  using I = spblas::index_t;

  for (auto&& [m, n, nnz] : util::dims) {
    // CSR arrays of the n x m matrix A^T are the CSC arrays of the m x n A.
    auto [values, colptr, rowind, shape_t, _] =
        spblas::generate_csr<T, I>(n, m, nnz);
    spblas::csc_wrapper<T, I> a(values, colptr, rowind, {I(m), I(n)}, nnz);

    T alpha = -1.5;
    auto x = util::make_values<T>(n);
    std::vector<T> y(m);

    spblas::operation_state state;
    spblas::multiply(state, alpha, a, x, y);

    std::vector<T> y_ref(m, T(0));
    for (std::size_t j = 0; j < n; j++) {
      for (auto p = colptr[j]; p < colptr[j + 1]; p++) {
        y_ref[rowind[p]] += alpha * (values[p] * x[j]);
      }
    }
    for (std::size_t i = 0; i < m; i++) {
      EXPECT_EQ_(y_ref[i], y[i]);
    }

    // The same through a matrix_opt of the CSC matrix.
    std::vector<T> y_opt(m);
    spblas::matrix_opt a_opt(a);
    spblas::multiply(state, alpha, a_opt, x, y_opt);
    for (std::size_t i = 0; i < m; i++) {
      EXPECT_EQ_(y_ref[i], y_opt[i]);
    }
  }
}

TEST(Multiply, SpMV_MatrixOptAndPolicy) {
  using T = float;
  using I = spblas::index_t;

  for (auto&& [m, n, nnz] : util::dims) {
    auto [values, rowptr, colind, shape, _] =
        spblas::generate_csr<T, I>(m, n, nnz);
    spblas::csr_wrapper<T, I> a(values, rowptr, colind, shape, nnz);
    spblas::matrix_opt a_opt(a);

    T alpha = 3, beta = 0.5;
    auto x = util::make_values<T>(n);
    auto z = util::make_values<T>(m, 1);
    auto y_ref = spmv_ref(values, rowptr, colind, m, x, alpha, beta, z);

    spblas::operation_state state;
    std::vector<T> y(m);

    spblas::multiply_inspect(util::test_policy{}, state, alpha, a_opt, x, beta,
                             z, y);
    spblas::multiply(util::test_policy{}, state, alpha, a_opt, x, beta, z, y);
    for (std::size_t i = 0; i < m; i++) {
      EXPECT_EQ_(y_ref[i], y[i]);
    }

    // Repeated execution with the same state.
    std::fill(y.begin(), y.end(), T(0));
    spblas::multiply(state, alpha, a_opt, x, beta, z, y);
    for (std::size_t i = 0; i < m; i++) {
      EXPECT_EQ_(y_ref[i], y[i]);
    }
  }
}

TEST(Multiply, SpMV_Int32Index) {
  using T = double;
  using I = std::int32_t;

  for (auto&& [m, n, nnz] : util::dims) {
    auto [values, rowptr, colind, shape, _] =
        spblas::generate_csr<T, I>(m, n, nnz);
    spblas::csr_wrapper<T, I> a(values, rowptr, colind, shape, nnz);

    auto x = util::make_values<T>(n);
    std::vector<T> y(m);

    spblas::operation_state state;
    spblas::multiply(state, T(2), a, x, y);

    auto y_ref = spmv_ref(values, rowptr, colind, m, x, T(2), T(0), y);
    for (std::size_t i = 0; i < m; i++) {
      EXPECT_EQ_(y_ref[i], y[i]);
    }
  }
}

TEST(Multiply, SpMV_Conjugated) {
  using T = std::complex<float>;
  using I = spblas::index_t;

  for (auto&& [m, n, nnz] : util::dims) {
    auto [values, rowptr, colind, shape, _] =
        spblas::generate_csr<T, I>(m, n, nnz);
    spblas::csr_wrapper<T, I> a(values, rowptr, colind, shape, nnz);

    T alpha(1, -2);
    std::vector<T> x(n, T(1, 1));
    std::vector<T> y(m);

    spblas::operation_state state;
    spblas::multiply(state, alpha, spblas::view::conjugate(a), x, y);

    std::vector<T> conj_values(values.size());
    std::transform(values.begin(), values.end(), conj_values.begin(),
                   [](T v) { return std::conj(v); });
    auto y_ref = spmv_ref(conj_values, rowptr, colind, m, x, alpha, T(0), y);
    for (std::size_t i = 0; i < m; i++) {
      EXPECT_EQ_(y_ref[i].real(), y[i].real());
      EXPECT_EQ_(y_ref[i].imag(), y[i].imag());
    }
  }
}

// A real conjugated vector operand is the vector itself, used in place.
TEST(Multiply, SpMV_ConjugateOfRealVector) {
  using T = double;
  using I = spblas::index_t;

  auto [m, n, nnz] = util::dims[1];
  auto [values, rowptr, colind, shape, _] =
      spblas::generate_csr<T, I>(m, n, nnz);
  spblas::csr_wrapper<T, I> a(values, rowptr, colind, shape, nnz);

  std::vector<T> x(n, T(0));
  auto x_conj = spblas::view::conjugate(x);
  // Filled after the view was made: the view must see these values.
  for (std::size_t j = 0; j < n; j++) {
    x[j] = T(j + 1);
  }
  std::vector<T> y(m), y_ref(m);

  spblas::operation_state state;
  spblas::multiply(state, T(2), a, x_conj, y);
  spblas::multiply(state, T(2), a, x, y_ref);
  for (std::size_t i = 0; i < m; i++) {
    EXPECT_EQ_(y_ref[i], y[i]);
  }
}

TEST(Multiply, SpMV_DimensionMismatchThrows) {
  using T = float;
  using I = spblas::index_t;

  auto [values, rowptr, colind, shape, nnz] =
      spblas::generate_csr<T, I>(10, 20, 30);
  spblas::csr_wrapper<T, I> a(values, rowptr, colind, shape, nnz);

  std::vector<T> x(20), y(10), z(11), x_bad(19);
  spblas::operation_state state;

  EXPECT_THROW(spblas::multiply(state, T(1), a, x_bad, y),
               std::invalid_argument);
  EXPECT_THROW(spblas::multiply(state, T(1), a, x, T(1), z, y),
               std::invalid_argument);
}

TEST(Multiply, SpMV_TransposedMatrixOpt) {
  using T = double;
  using I = spblas::index_t;

  for (auto&& [m, n, nnz] : util::dims) {
    auto [values, rowptr, colind, shape, _] =
        spblas::generate_csr<T, I>(m, n, nnz);
    spblas::csr_wrapper<T, I> a(values, rowptr, colind, shape, nnz);
    spblas::matrix_opt a_opt(a);

    // y = alpha * A^T * x
    T alpha = -2;
    auto x = util::make_values<T>(m);
    std::vector<T> y(n);
    spblas::operation_state state;
    spblas::multiply(state, alpha, spblas::view::transpose(a_opt), x, y);

    std::vector<T> y_ref(n, T(0));
    for (std::size_t i = 0; i < m; i++) {
      for (auto p = rowptr[i]; p < rowptr[i + 1]; p++) {
        y_ref[colind[p]] += alpha * (values[p] * x[i]);
      }
    }
    for (std::size_t j = 0; j < n; j++) {
      EXPECT_EQ_(y_ref[j], y[j]);
    }
  }
}

TEST(Multiply, SpMV_Int64Index) {
  // std::int64_t is `long`, not MKL_INT64 (`long long`): exercises index
  // dispatch by size in vendor backends.
  using T = double;
  using I = std::int64_t;

  for (auto&& [m, n, nnz] : util::dims) {
    auto [values, rowptr, colind, shape, _] =
        spblas::generate_csr<T, I>(m, n, nnz);
    spblas::csr_wrapper<T, I> a(values, rowptr, colind, shape, nnz);
    spblas::matrix_opt a_opt(a);

    auto x = util::make_values<T>(n);
    std::vector<T> y(m), y_opt(m);
    spblas::operation_state state;
    spblas::multiply(state, T(2), a, x, y);
    spblas::multiply_inspect(state, T(2), a_opt, x, y_opt);
    spblas::multiply(state, T(2), a_opt, x, y_opt);

    auto y_ref = spmv_ref(values, rowptr, colind, m, x, T(2), T(0), y);
    for (std::size_t i = 0; i < m; i++) {
      EXPECT_EQ_(y_ref[i], y[i]);
      EXPECT_EQ_(y_ref[i], y_opt[i]);
    }
  }
}

TEST(Multiply, SpMV_ScaleInvalidatesMatrixOpt) {
  using T = double;
  using I = spblas::index_t;

  auto [m, n, nnz] = util::dims[1];
  auto [values, rowptr, colind, shape, _] =
      spblas::generate_csr<T, I>(m, n, nnz);
  spblas::csr_wrapper<T, I> a(values, rowptr, colind, shape, nnz);
  spblas::matrix_opt a_opt(a);

  auto x = util::make_values<T>(n);
  std::vector<T> y(m), y_scaled(m);
  spblas::operation_state state;
  spblas::multiply_inspect(state, T(1), a_opt, x, y);
  spblas::multiply(state, T(1), a_opt, x, y);

  // Changing the values through the library keeps the matrix_opt
  // consistent (optimized data derived from the old values is dropped).
  spblas::scale(state, T(3), a_opt);
  spblas::multiply(state, T(1), a_opt, x, y_scaled);
  for (std::size_t i = 0; i < m; i++) {
    EXPECT_EQ_(T(3) * y[i], y_scaled[i]);
  }
}

TEST(Multiply, SpMV_ConjugatedMatrixOptBeta) {
  using T = std::complex<double>;
  using I = spblas::index_t;

  for (auto&& [m, n, nnz] : util::dims) {
    auto [values, rowptr, colind, shape, _] =
        spblas::generate_csr<T, I>(m, n, nnz);
    spblas::csr_wrapper<T, I> a(values, rowptr, colind, shape, nnz);
    spblas::matrix_opt a_opt(a);

    T alpha(2, -1), beta(0.5, 1);
    std::vector<T> x(n), z(m);
    for (std::size_t j = 0; j < n; j++) {
      x[j] = T(double(j % 5), double(j % 3) - 1);
    }
    for (std::size_t i = 0; i < m; i++) {
      z[i] = T(1, double(i % 4));
    }

    std::vector<T> conj_values(values.size());
    std::transform(values.begin(), values.end(), conj_values.begin(),
                   [](T v) { return std::conj(v); });
    auto y_ref = spmv_ref(conj_values, rowptr, colind, m, x, alpha, beta, z);

    // Inspected (optimized) handle, used for conj(A) in both a separate and
    // an aliased output.
    spblas::operation_state state;
    std::vector<T> y(m), y_alias = z;
    auto a_conj = spblas::view::conjugate(a_opt);
    spblas::multiply_inspect(state, alpha, a_conj, x, beta, z, y);
    spblas::multiply(state, alpha, a_conj, x, beta, z, y);
    spblas::multiply(state, alpha, a_conj, x, beta, y_alias, y_alias);

    for (std::size_t i = 0; i < m; i++) {
      EXPECT_NEAR(y_ref[i].real(), y[i].real(),
                  1e-9 * (1 + std::abs(y_ref[i])));
      EXPECT_NEAR(y_ref[i].imag(), y[i].imag(),
                  1e-9 * (1 + std::abs(y_ref[i])));
      EXPECT_NEAR(y_ref[i].real(), y_alias[i].real(),
                  1e-9 * (1 + std::abs(y_ref[i])));
      EXPECT_NEAR(y_ref[i].imag(), y_alias[i].imag(),
                  1e-9 * (1 + std::abs(y_ref[i])));
    }
  }
}

TEST(Multiply, SpMV_UpdateMatrixAfterInPlaceChange) {
  using T = double;
  using I = spblas::index_t;

  auto [m, n, nnz] = util::dims[1];
  auto [values, rowptr, colind, shape, _] =
      spblas::generate_csr<T, I>(m, n, nnz);
  spblas::csr_wrapper<T, I> a(values, rowptr, colind, shape, nnz);
  spblas::matrix_opt a_opt(a);

  auto x = util::make_values<T>(n);
  std::vector<T> y(m), y_changed(m);
  spblas::operation_state state;
  spblas::multiply_inspect(state, T(1), a_opt, x, y);
  spblas::multiply(state, T(1), a_opt, x, y);

  // The user changes the values in place and tells the library.
  for (auto&& v : values) {
    v *= T(-2);
  }
  spblas::update_matrix(a_opt);
  spblas::multiply(state, T(1), a_opt, x, y_changed);
  for (std::size_t i = 0; i < m; i++) {
    EXPECT_EQ_(T(-2) * y[i], y_changed[i]);
  }
}
