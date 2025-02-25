#pragma once

#include <spblas/concepts.hpp>
#include <spblas/detail/operation_info_t.hpp>

template <class ExecutionPolicy, in - matrix InMat, class Triangle,
          class DiagonalStorage, in - vector InVec, out - vector OutVec>
void triangular_matrix_vector_solve(ExecutionPolicy&& exec, InMat A, Triangle t,
                                    DiagonalStorage d, InVec b, OutVec x);

namespace spblas {

template <matrix A, class Triangle, class DiagonalStorage, vector B, vector X>
void triangular_solve(A&& a, Triangle uplo, DiagonalStorage diag, B&& b, X&& x);

} // namespace spblas
