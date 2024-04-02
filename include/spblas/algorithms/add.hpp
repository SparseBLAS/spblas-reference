#pragma once

#include <spblas/concepts.hpp>
#include <spblas/detail/operation_info_t.hpp>

namespace spblas {

template <matrix A, matrix B, matrix C>
void add(A&& a, B&& b, C&& c);

template <vector A, vector B, vector C>
void add(A&& a, B&& b, C&& c);

} // namespace spblas
