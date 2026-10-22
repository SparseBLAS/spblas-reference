#pragma once

#include <stdexcept>

#include <spblas/algorithms/spmv_impl.hpp>
#include <spblas/backend/backend.hpp>
#include <spblas/concepts.hpp>
#include <spblas/detail/api_concepts.hpp>
#include <spblas/detail/log.hpp>
#include <spblas/detail/operation_state.hpp>
#include <spblas/detail/view_analysis.hpp>

//
// Reference SpMV with structure views: y = alpha * A * x [+ beta * z]
//
//   multiply([policy,] state, alpha, A, x, [beta, z,] y)
//
//   A  view::triangle (SpTRMV), view::diagonal, view::symmetric (SpSYMV) or
//      view::hermitian (SpHEMV) of CSR / CSC / matrix_opt, composed with
//      view::transpose / view::conjugate in any order. The stack is reduced
//      by __detail::analyze_view, and one pass over the stored entries of
//      the base matrix applies every represented entry (mirrored entries of
//      symmetric / hermitian views and implied unit diagonals included).
//
//   The other overloads (no beta / z, inspect, policy) are the general ones
//   in spmv_impl.hpp / policy_forwarding.hpp, which forward here.
//

namespace spblas {

template <__detail::scalar Alpha, matrix A, vector X, __detail::scalar Beta,
          vector Z, vector Y>
  requires(__detail::structured_view<A> && __backend::lookupable<X> &&
           __backend::lookupable<Z> && __backend::lookupable<Y>)
void multiply(operation_state& state, Alpha alpha, A&& a, X&& x, Beta beta,
              Z&& z, Y&& y) {
  log_trace("");
  auto analysis = __detail::analyze_view(a);
  const auto shape = analysis.shape();
  if (shape[0] != __backend::shape(y) || shape[1] != __backend::shape(x) ||
      __backend::shape(z) != __backend::shape(y)) {
    throw std::invalid_argument(
        "multiply: matrix and vector dimensions are incompatible.");
  }

  using V = tensor_scalar_t<Y>;
  const V alpha_v = static_cast<V>(alpha);

  __detail::assign_scaled_(beta, z, y);

  __detail::for_each_represented_entry(
      analysis, [&](auto i, auto k, auto&& a_v) {
        __backend::lookup(y, i) += alpha_v * (a_v * __backend::lookup(x, k));
      });
}

} // namespace spblas
