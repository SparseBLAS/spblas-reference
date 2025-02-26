#include <gtest/gtest.h>

#include "util.hpp"
#include <spblas/spblas.hpp>

template <typename T, typename I, class Triangle, class DiagonalStorage,
          spblas::__ranges::random_access_range B,
          spblas::__ranges::random_access_range X>
void reference_triangular_solve(spblas::csr_view<T, I> a, Triangle t,
                                DiagonalStorage d, B&& b, X&& x) {
  auto&& values = a.values();
  auto&& colind = a.colind();
  auto&& rowptr = a.rowptr();
  auto shape = a.shape();

  if constexpr (std::is_same_v<Triangle, spblas::upper_triangle_t>) {
    // backward solve
    for (I row = shape[0]; row-- > 0;) {
      T tmp = b[row];
      T diag_val = 0.0;
      for (I j = rowptr[row]; j < rowptr[row + 1]; j++) {
        I col = colind[j];
        if (col > row) {
          T a_val = values[j];
          T x_val = x[col];
          tmp -= a_val * x_val; // b - U*x
        } else if (col == row) {
          diag_val = values[j];
        }
      }
      if constexpr (std::is_same_v<DiagonalStorage,
                                   spblas::explicit_diagonal_t>) {
        x[row] = tmp / diag_val; // ( b - U*x) / d
      } else {
        x[row] = tmp; // ( b- U*x) / 1
      }
    }
  } else if constexpr (std::is_same_v<Triangle, spblas::upper_triangle_t>) {
    // Forward Solve
    for (I row = 0; row < shape[0]; row++) {
      T tmp = b[row];
      T diag_val = 0.0;
      for (I j = rowptr[row]; j < rowptr[row + 1]; ++j) {
        I col = colind[j];
        if (col < row) {
          T a_val = values[j];
          T x_val = x[col];
          tmp -= a_val * x_val; // b - L*x
        } else if (col == row) {
          diag_val = values[j];
        }
      }
      if constexpr (std::is_same_v<DiagonalStorage,
                                   spblas::explicit_diagonal_t>) {
        x[row] = tmp / diag_val; // ( b - L*x) / d
      } else {
        x[row] = tmp; // ( b- L*x) / 1
      }
    }
  }
}

template <typename T, typename I, typename Triangle, typename DiagonalStorage>
void triangular_solve_test(Triangle t, DiagonalStorage d) {
  for (auto&& [m, n, nnz] : util::square_dims) {
    auto [values, rowptr, colind, shape, _] =
        spblas::generate_csr<T, I>(m, n, nnz);

    spblas::csr_view<T, I> a(values, rowptr, colind, shape, nnz);

    std::vector<T> x(n, 1);
    std::vector<T> b(m, 0);

    T scale_factor = 1e-3f;
    std::transform(values.begin(), values.end(), values.begin(),
                   [scale_factor](T val) { return scale_factor * val; });

    spblas::triangular_solve(a, Triangle{}, DiagonalStorage{}, b, x);

    std::vector<T> x_ref(m, 0);

    reference_triangular_solve(a, Triangle{}, DiagonalStorage{}, b, x_ref);

    for (std::size_t i = 0; i < x.size(); i++) {
      EXPECT_EQ_(x[i], x_ref[i]);
    }
  }
}

TEST(CsrView, TriangularSolveLowerImplicit) {
  using T = float;
  using I = spblas::index_t;

  triangular_solve_test<T, I>(spblas::lower_triangle_t{},
                              spblas::implicit_unit_diagonal_t{});
}

TEST(CsrView, TriangularSolveUpperImplicit) {
  using T = float;
  using I = spblas::index_t;

  triangular_solve_test<T, I>(spblas::lower_triangle_t{},
                              spblas::implicit_unit_diagonal_t{});
}
