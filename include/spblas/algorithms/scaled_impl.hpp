#pragma once

#include <spblas/concepts.hpp>

namespace spblas {

template <typename Scalar, vector V>
auto scaled(Scalar alpha, V&& v) {
  return __ranges::views::transform(std::forward<V>(v),
                                    [=](auto&& x) { return alpha * x; });
}

} // namespace spblas
