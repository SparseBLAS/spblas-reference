#pragma once

#include <spblas/backend/backend.hpp>
#include <spblas/concepts.hpp>

#include <algorithm>

namespace {

// template <typename T>
// __global__ scaled_vector(int n, T scalar, T* vec) {
//   auto id = threadIdx.x + blockDim.x * blockIdx.x;
//   if (id < n) {
//     vec[id] *= scalar;
//   }
// }

template <typename I>
I ceildiv(I a, I b) {
  return (a + b - 1) / b;
}
} // namespace

namespace spblas {

template <typename Scalar, matrix M>
void scale(Scalar alpha, M&& m) {
  auto value = m.values().data();
  auto size = m.values().size();
  // scaled_vector<<<ceildiv(size, 256), 256>>>(size, alpha, value);
}

template <typename Scalar, vector V>
void scale(Scalar alpha, V&& v) {
  auto value = v.data_handle();
  auto size = v.size();
  // scaled_vector<<<ceildiv(size, 256), 256>>>(size, alpha, value);
}

} // namespace spblas
