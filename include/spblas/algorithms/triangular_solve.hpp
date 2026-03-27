#pragma once

#include <spblas/concepts.hpp>
#include <spblas/detail/operation_info_t.hpp>

namespace spblas {


template <matrix A, class Triangle, class DiagonalStorage, vector B, vector X>
void triangular_solve_inspect(operation_info_t& info, A&& a, Triangle uplo, DiagonalStorage diag, B&& b, X&& x);


template <matrix A, class Triangle, class DiagonalStorage, vector B, vector X>
operation_info_t triangular_solve_inspect(A&& a, Triangle uplo, DiagonalStorage diag, B&& b, X&& x);


template <matrix A, class Triangle, class DiagonalStorage, vector B, vector X>
void triangular_solve(A&& a, Triangle uplo, DiagonalStorage diag, B&& b, X&& x);

} // namespace spblas
