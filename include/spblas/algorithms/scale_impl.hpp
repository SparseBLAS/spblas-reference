#pragma once

#include <spblas/backend/backend.hpp>
#include <spblas/concepts.hpp>
#include <spblas/detail/api_concepts.hpp>
#include <spblas/detail/operation_state.hpp>

#include <algorithm>

namespace spblas {

namespace {

template <typename Scalar, typename T>
  requires(matrix<T> || vector<T>)
void scale_impl_(Scalar alpha, T&& t) {
  auto&& values = __backend::values(t);
  std::for_each(__ranges::begin(values), __ranges::end(values),
                [&](auto&& v) { v *= alpha; });
}

} // namespace

template <typename Scalar, matrix M>
void scale(Scalar alpha, M&& m) {
  scale_impl_(alpha, std::forward<M>(m));
}

template <typename Scalar, vector V>
void scale(Scalar alpha, V&& v) {
  scale_impl_(alpha, std::forward<V>(v));
}

//
// operation_state API (SparseBLAS paper)
//
//   scale([policy,] state, alpha, A)    A := alpha * A, in place
//

template <__detail::scalar Alpha, matrix M>
  requires(__detail::is_csr_wrapper_v<M> || __detail::is_csc_wrapper_v<M> ||
           __detail::is_matrix_opt_v<M>)
void scale(operation_state& state, Alpha alpha, M&& m) {
  if constexpr (__detail::is_matrix_opt_v<M>) {
    scale_impl_(alpha, m.base());
    // The values changed: drop optimization data derived from them.
    m.state_->invalidate();
  } else {
    scale_impl_(alpha, std::forward<M>(m));
  }
}

// Optional leading execution policy. The reference implementation runs
// sequentially and ignores it.
template <__detail::execution_policy P, typename... Args>
void scale(P&& policy, operation_state& state, Args&&... args) {
  scale(state, std::forward<Args>(args)...);
}

} // namespace spblas
