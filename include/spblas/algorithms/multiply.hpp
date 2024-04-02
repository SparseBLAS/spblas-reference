#pragma once

#include <spblas/concepts.hpp>
#include <spblas/detail/operation_info_t.hpp>

namespace spblas {

template <matrix A, vector B, vector C>
void multiply(A&& a, B&& b, C&& c);

template <matrix A, matrix B, matrix C>
void multiply(A&& a, B&& b, C&& c);

template <matrix A, matrix B, matrix C>
operation_info_t multiply_inspect(A&& a, B&& b, C&& c);

template <matrix A, matrix B, matrix C>
void multiply_execute(operation_info_t& info, A&& a, B&& b, C&& c);

} // namespace spblas
