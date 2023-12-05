#pragma once

#include <spblas/concepts.hpp>

namespace spblas {

template <typename Scalar, matrix M>
auto scaled(Scalar alpha, M&& m);

template <typename Scalar, vector V>
auto scaled(Scalar alpha, V&& v);

} // namespace spblas
