#pragma once

#include <type_traits>
#include <utility>

#include <spblas/algorithms/conjugated_impl.hpp>
#include <spblas/algorithms/transposed.hpp>
#include <spblas/concepts.hpp>
#include <spblas/views/inspectors.hpp>
#include <spblas/views/transposed_view.hpp>
#include <spblas/views/triangle_view.hpp>

// View functions from the SparseBLAS paper (`spblas::view::...`). Views are
// lightweight and non-owning; implementations may still materialize them
// (e.g. an explicit transpose inside a matrix_opt).
//
// Not yet provided: view::diagonal, view::symmetric, view::hermitian.

namespace spblas {

namespace view {

// Transposed view of `m`:
//   - CSR / CSC wrapper: the same arrays reinterpreted as CSC / CSR;
//   - transposed_view: the original matrix;
//   - anything else (matrix_opt, conjugated views, dense matrices): a
//     transposed_view, which keeps e.g. a matrix_opt's optimization data.
template <matrix M>
auto transpose(M&& m) {
  if constexpr (__detail::is_csr_wrapper_v<M> ||
                __detail::is_csc_wrapper_v<M>) {
    return __detail::reinterpret_transposed(std::forward<M>(m));
  } else if constexpr (__detail::is_transposed_view_v<M>) {
    return m.matrix();
  } else {
    return transposed_view<std::remove_cvref_t<M>>(std::forward<M>(m));
  }
}

// Conjugated view of `t`; `t` itself for non-complex value types.
template <tensor T>
auto conjugate(T&& t) {
  return conjugated(std::forward<T>(t));
}

// The `u` triangle of `m` with diagonal handling `d`.
template <matrix M>
auto triangle(M&& m, uplo u, diag d = diag::explicit_diagonal) {
  return triangle_view<std::remove_cvref_t<M>>(std::forward<M>(m), u, d);
}

} // namespace view

} // namespace spblas
