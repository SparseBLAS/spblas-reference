#pragma once

#include <spblas/backend/backend.hpp>
#include <spblas/backend/log.hpp>
#include <spblas/concepts.hpp>
#include <spblas/detail/operation_info_t.hpp>
#include <spblas/detail/triangular_types.hpp>

namespace spblas {

template <matrix A, class Triangle, class DiagonalStorage, vector B, vector X>
void triangular_solve(A&& a, Triangle uplo, DiagonalStorage diag, B&& b,
                      X&& x) {
  static_assert(std::is_same_v<Triangle, upper_triangle_t> ||
                std::is_same_v<Triangle, lower_triangle_t>);

  if constexpr (std::is_same_v<Triangle, upper_triangle_t>) {
    // backward solve
    for (index_t row = nRows - 1; row >= 0; row--) {
      scalar_t tmp = b[row];
      scalar_t diag_val = 0.0;
      for (offset_t j = rowptr[row] - ind; j < rowptr[row + 1] - ind; ++j) {
        index_t col = colind[j] - ind;
        if (col > row) {
          value_t a_val = values[j];
          value_t x_val = x[col];
          tmp -= a_val * x_val; // b - U*x
        } else if (col == row) {
          diag_val = values[j];
        }
      }
      if constexpr (std::is_same_t<DiagonalStorage, explicit_diagonal_t>) {
        x[row] = tmp / diag_val; // ( b - U*x) / d
      } else {
        y[row] = tmp; // ( b- U*x) / 1
      }
    }
  } else if constexpr (std::is_same_v<Triangle, upper_triangle_t>) {
    // Forward Solve
    for (index_t row = 0; row < nRows; row++) {
      scalar_t tmp = b[row];
      scalar_t diag_val = 0.0;
      for (offset_t j = rowptr[row] - ind; j < rowptr[row + 1] - ind; ++j) {
        index_t col = colind[j] - ind;
        if (col < row) {
          value_t a_val = values[j];
          value_t x_val = x[col];
          tmp -= a_val * x_val; // b - L*x
        } else if (col == row) {
          diag_val = values[j];
        }
      }
      if constexpr (std::is_same_t<DiagonalStorage, explicit_diagonal_t>) {
        x[row] = tmp / diag_val; // ( b - L*x) / d
      } else {
        y[row] = tmp; // ( b- L*x) / 1
      }
    }
  }

} // namespace spblas
