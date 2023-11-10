#pragma once

#include <spblas/concepts.hpp>
#include <spblas/backend/backend.hpp>

#include <algorithm>

namespace spblas {

namespace {

template <typename Scalar, typename T>
requires(matrix<T> || vector<T>)
void scale_impl_(Scalar alpha, T&& t) {
  auto&& values = __backend::values(t);
  std::for_each(__ranges::begin(values), __ranges::end(values),
                [&](auto&& v) {
                  v *= alpha;
                });
}

}

template <typename Scalar, matrix M>
void scale(Scalar alpha, M&& m) {
  scale_impl_(alpha, std::forward<M>(m));
}

template <typename Scalar, vector V>
void scale(Scalar alpha, V&& v) {
  scale_impl_(alpha, std::forward<V>(v));
}

} // namespace spblas