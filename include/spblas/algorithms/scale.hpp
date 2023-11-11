#pragma once

#include <spblas/concepts.hpp>

namespace spblas {

template <typename Scalar, matrix M>
void scale(Scalar alpha, M&& m);

template <typename Scalar, vector V>
void scale(Scalar alpha, V&& v);

} // namespace spblas
