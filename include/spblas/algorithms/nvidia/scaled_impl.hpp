#pragma once

#include <spblas/concepts.hpp>

namespace spblas {

template <typename Scalar, vector V>
auto scaled(Scalar alpha, V&& v) {
  return std::pair{alpha, v};
}

} // namespace spblas
