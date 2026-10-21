#pragma once

#include <stdexcept>

#include <spblas/backend/backend.hpp>
#include <spblas/concepts.hpp>
#include <spblas/detail/api_concepts.hpp>
#include <spblas/detail/log.hpp>
#include <spblas/detail/operation_state.hpp>

//
// Reference SpMM: Y = alpha * A * X [+ beta * Z]
//
//   multiply_inspect([policy,] state, alpha, A, X, [beta, Z,] Y)
//   multiply([policy,] state, alpha, A, X, [beta, Z,] Y)
//
//   A        CSR, CSC, a matrix_opt of either, and view::transpose /
//            view::conjugate of those. One kernel (a for_each over the stored
//            entries) serves all of them.
//   X, Y, Z  dense matrices (mdspan, any layout with element lookup).
//            Z may alias Y; if beta == 0, Z is not read.
//
//   Not yet supported: view::triangle / symmetric / hermitian of A, and a
//   dense A with a sparse X.
//
// The policy-first overloads are in multiply_impl.hpp.
//

namespace spblas {

namespace __detail {

// Y = beta * Z, or Y = 0 if beta == 0 (Z is not read then).
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

// Y = alpha * A * X + beta * Z
template <__detail::scalar Alpha, matrix A, matrix X, __detail::scalar Beta,
          matrix Z, matrix Y>
  requires(__backend::lookupable<X> && __backend::lookupable<Z> &&
           __backend::lookupable<Y>)
void multiply(operation_state& state, Alpha alpha, A&& a, X&& x, Beta beta,
              Z&& z, Y&& y) {
  log_trace("");
  if (__backend::shape(a)[0] != __backend::shape(y)[0] ||
      __backend::shape(x)[1] != __backend::shape(y)[1] ||
      __backend::shape(a)[1] != __backend::shape(x)[0] ||
      __backend::shape(z)[0] != __backend::shape(y)[0] ||
      __backend::shape(z)[1] != __backend::shape(y)[1]) {
    throw std::invalid_argument(
        "multiply: matrix dimensions are incompatible.");
  }

  using V = tensor_scalar_t<Y>;
  const V alpha_v = static_cast<V>(alpha);

  __detail::assign_scaled_(beta, z, y);

  __backend::for_each(a, [&](auto&& e) {
    auto&& [idx, a_v] = e;
    auto&& [i, k] = idx;
    for (tensor_index_t<Y> j = 0; j < __backend::shape(x)[1]; j++) {
      __backend::lookup(y, i, j) +=
          alpha_v * (a_v * __backend::lookup(x, k, j));
    }
  });
}

// Y = alpha * A * X
template <__detail::scalar Alpha, matrix A, matrix X, matrix Y>
  requires(__backend::lookupable<X> && __backend::lookupable<Y>)
void multiply(operation_state& state, Alpha alpha, A&& a, X&& x, Y&& y) {
  log_trace("");
  multiply(state, alpha, std::forward<A>(a), std::forward<X>(x),
           tensor_scalar_t<Y>(0), y, y);
}

// The reference implementation has nothing to prepare.
template <__detail::scalar Alpha, matrix A, matrix X, matrix Y>
  requires(__backend::lookupable<X> && __backend::lookupable<Y>)
void multiply_inspect(operation_state& state, Alpha alpha, A&& a, X&& x,
                      Y&& y) {
  log_trace("");
}

template <__detail::scalar Alpha, matrix A, matrix X, __detail::scalar Beta,
          matrix Z, matrix Y>
  requires(__backend::lookupable<X> && __backend::lookupable<Z> &&
           __backend::lookupable<Y>)
void multiply_inspect(operation_state& state, Alpha alpha, A&& a, X&& x,
                      Beta beta, Z&& z, Y&& y) {
  log_trace("");
}

} // namespace spblas
