#pragma once

#include <spblas/backend/backend.hpp>
#include <spblas/concepts.hpp>

namespace spblas {

namespace __detail {

// y = beta * z, or y = 0 if beta == 0 (z is not read then). z may alias y.
template <typename Beta, vector Z, vector Y>
void assign_scaled_(Beta beta, Z&& z, Y&& y) {
  using V = tensor_scalar_t<Y>;
  const V beta_v = static_cast<V>(beta);
  const bool zero = beta_v == V(0);

  for (tensor_index_t<Y> i = 0; i < __backend::shape(y); i++) {
    __backend::lookup(y, i) = zero ? V(0) : beta_v * __backend::lookup(z, i);
  }
}

// Y = beta * Z, or Y = 0 if beta == 0 (Z is not read then). Z may alias Y.
template <typename Beta, matrix Z, matrix Y>
void assign_scaled_(Beta beta, Z&& z, Y&& y) {
  using V = tensor_scalar_t<Y>;
  const V beta_v = static_cast<V>(beta);
  const bool zero = beta_v == V(0);

  for (tensor_index_t<Y> i = 0; i < __backend::shape(y)[0]; i++) {
    for (tensor_index_t<Y> j = 0; j < __backend::shape(y)[1]; j++) {
      __backend::lookup(y, i, j) =
          zero ? V(0) : beta_v * __backend::lookup(z, i, j);
    }
  }
}

} // namespace __detail

} // namespace spblas
