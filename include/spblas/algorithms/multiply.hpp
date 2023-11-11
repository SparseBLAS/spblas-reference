#pragma once

#include <spblas/concepts.hpp>

namespace spblas {

template <matrix A, vector B, vector C>
void multiply(A&& a, B&& b, C&& c);

} // namespace spblas
