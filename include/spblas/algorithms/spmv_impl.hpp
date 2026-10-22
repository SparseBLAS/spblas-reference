#pragma once

#include <stdexcept>

#include <spblas/backend/backend.hpp>
#include <spblas/concepts.hpp>
#include <spblas/detail/api_concepts.hpp>
#include <spblas/detail/log.hpp>
#include <spblas/detail/operation_state.hpp>

//
// Reference SpMV: y = alpha * A * x [+ beta * z]
//
//   multiply_inspect([policy,] state, alpha, A, x, [beta, z,] y)
//   multiply([policy,] state, alpha, A, x, [beta, z,] y)
//
//   A        CSR, CSC, a matrix_opt of either, and view::transpose /
//            view::conjugate of those. One kernel (a for_each over the stored
//            entries) serves all of them.
//   x, y, z  dense vectors. z may alias y; if beta == 0, z is not read.
//
//   Not yet supported: view::triangle / symmetric / hermitian of A.
//
// The policy-first overloads are in policy_forwarding.hpp.
//

namespace spblas {

namespace __detail {

// y = beta * z, or y = 0 if beta == 0 (z is not read then).
template <typename Beta, vector Z, vector Y>
void assign_scaled_(Beta beta, Z&& z, Y&& y) {
  using V = tensor_scalar_t<Y>;
  const V beta_v = static_cast<V>(beta);
  const bool zero = beta_v == V(0);

  for (tensor_index_t<Y> i = 0; i < __backend::shape(y); i++) {
    __backend::lookup(y, i) = zero ? V(0) : beta_v * __backend::lookup(z, i);
  }
}

} // namespace __detail

// y = alpha * A * x + beta * z
template <__detail::scalar Alpha, matrix A, vector X, __detail::scalar Beta,
          vector Z, vector Y>
  requires(__backend::lookupable<X> && __backend::lookupable<Z> &&
           __backend::lookupable<Y>)
void multiply(operation_state& state, Alpha alpha, A&& a, X&& x, Beta beta,
              Z&& z, Y&& y) {
  log_trace("");
  if (__backend::shape(a)[0] != __backend::shape(y) ||
      __backend::shape(a)[1] != __backend::shape(x) ||
      __backend::shape(z) != __backend::shape(y)) {
    throw std::invalid_argument(
        "multiply: matrix and vector dimensions are incompatible.");
  }

  using V = tensor_scalar_t<Y>;
  const V alpha_v = static_cast<V>(alpha);

  __detail::assign_scaled_(beta, z, y);

  __backend::for_each(a, [&](auto&& e) {
    auto&& [idx, a_v] = e;
    auto&& [i, k] = idx;
    __backend::lookup(y, i) += alpha_v * (a_v * __backend::lookup(x, k));
  });
}

// y = alpha * A * x
template <__detail::scalar Alpha, matrix A, vector X, vector Y>
  requires(__backend::lookupable<X> && __backend::lookupable<Y>)
void multiply(operation_state& state, Alpha alpha, A&& a, X&& x, Y&& y) {
  log_trace("");
  multiply(state, alpha, std::forward<A>(a), std::forward<X>(x),
           tensor_scalar_t<Y>(0), y, y);
}

// The reference implementation has nothing to prepare.
template <__detail::scalar Alpha, matrix A, vector X, vector Y>
void multiply_inspect(operation_state& state, Alpha alpha, A&& a, X&& x,
                      Y&& y) {
  log_trace("");
}

template <__detail::scalar Alpha, matrix A, vector X, __detail::scalar Beta,
          vector Z, vector Y>
void multiply_inspect(operation_state& state, Alpha alpha, A&& a, X&& x,
                      Beta beta, Z&& z, Y&& y) {
  log_trace("");
}

} // namespace spblas
