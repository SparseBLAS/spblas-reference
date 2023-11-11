#pragma once

#include <spblas/backend/backend.hpp>
#include <spblas/concepts.hpp>

#include <algorithm>

namespace spblas {

template <matrix A, vector B, vector C>
void multiply(A&& a, B&& b, C&& c) {
  // TODO: throw exception
  if (__backend::shape(a)[0] != __backend::shape(c) ||
      __backend::shape(a)[1] != __backend::shape(b)) {
    throw std::invalid_argument(
        "multiply: matrix and vector dimensions are incompatible.");
  }

  __backend::for_each(a, [&](auto&& e) {
    auto&& [idx, a_v] = e;
    auto&& [i, k] = idx;
    c[i] += a_v * b[k];
  });
}

} // namespace spblas
