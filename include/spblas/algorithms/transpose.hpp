#pragma once

#include <spblas/concepts.hpp>
#include <spblas/detail/operation_info_t.hpp>

namespace spblas {

template <matrix A, matrix B>
operation_info_t transpose_inspect(A&& a, B&& b);

template <matrix A, matrix B>
void transpose(operation_info_t& info, A&& a, B&& b);

} // namespace spblas
