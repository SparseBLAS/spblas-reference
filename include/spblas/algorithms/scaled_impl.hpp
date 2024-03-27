#pragma once

#include <spblas/concepts.hpp>
#include <spblas/views/scaled_view.hpp>

namespace spblas {

template <typename Scalar, vector V>
auto scaled(Scalar alpha, V&& v) {
  return scaled_view(alpha, std::forward<V>(v));
}

template <typename Scalar, matrix M>
auto scaled(Scalar alpha, M&& m) {
  return scaled_view(alpha, std::forward<M>(m));
}

} // namespace spblas
