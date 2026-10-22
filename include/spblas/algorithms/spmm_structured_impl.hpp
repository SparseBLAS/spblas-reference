#pragma once

#include <stdexcept>

#include <spblas/algorithms/spmm_impl.hpp>
#include <spblas/backend/backend.hpp>
#include <spblas/concepts.hpp>
#include <spblas/detail/api_concepts.hpp>
#include <spblas/detail/log.hpp>
#include <spblas/detail/operation_state.hpp>
#include <spblas/detail/view_analysis.hpp>

//
// Reference SpMM with structure views: Y = alpha * A * X [+ beta * Z]
//
//   multiply([policy,] state, alpha, A, X, [beta, Z,] Y)
//
//   A  view::triangle, view::diagonal, view::symmetric or view::hermitian of
//      CSR / CSC / matrix_opt, composed with view::transpose /
//      view::conjugate in any order (see spmv_structured_impl.hpp).
//   X, Y, Z dense matrices as for spmm_impl.hpp.
//

namespace spblas {

template <__detail::scalar Alpha, matrix A, matrix X, __detail::scalar Beta,
          matrix Z, matrix Y>
  requires(__detail::structured_view<A> && __backend::lookupable<X> &&
           __backend::lookupable<Z> && __backend::lookupable<Y>)
void multiply(operation_state& state, Alpha alpha, A&& a, X&& x, Beta beta,
              Z&& z, Y&& y) {
  log_trace("");
  auto analysis = __detail::analyze_view(a);
  const auto shape = analysis.shape();
  if (shape[0] != __backend::shape(y)[0] ||
      __backend::shape(x)[1] != __backend::shape(y)[1] ||
      shape[1] != __backend::shape(x)[0] ||
      __backend::shape(z)[0] != __backend::shape(y)[0] ||
      __backend::shape(z)[1] != __backend::shape(y)[1]) {
    throw std::invalid_argument(
        "multiply: matrix dimensions are incompatible.");
  }

  using V = tensor_scalar_t<Y>;
  const V alpha_v = static_cast<V>(alpha);

  __detail::assign_scaled_(beta, z, y);

  __detail::for_each_represented_entry(
      analysis, [&](auto i, auto k, auto&& a_v) {
        for (tensor_index_t<Y> j = 0; j < __backend::shape(x)[1]; j++) {
          __backend::lookup(y, i, j) +=
              alpha_v * (a_v * __backend::lookup(x, k, j));
        }
      });
}

} // namespace spblas
