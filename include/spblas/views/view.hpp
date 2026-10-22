#pragma once

#include <type_traits>
#include <utility>

#include <spblas/algorithms/conjugated_impl.hpp>
#include <spblas/algorithms/transposed.hpp>
#include <spblas/concepts.hpp>
#include <spblas/views/diagonal_view.hpp>
#include <spblas/views/inspectors.hpp>
#include <spblas/views/transposed_view.hpp>
#include <spblas/views/triangle_view.hpp>

// View functions from the SparseBLAS paper (`spblas::view::...`). Views are
// lightweight and non-owning; implementations may still materialize them
// (e.g. an explicit transpose inside a matrix_opt).
//
// Stacks of these views are reduced to one canonical form by
// __detail::analyze_view (detail/view_analysis.hpp).
//

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
    return transposed(std::forward<M>(m));
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
template <matrix M, uplo_tag U, diag_tag D = diag::explicit_diagonal_t>
auto triangle(M&& m, U u, D d = {}) {
  return triangle_view<std::remove_cvref_t<M>, U, D>(std::forward<M>(m));
}

// The diagonal of `m` (neither strict triangle) with diagonal handling `d`.
template <matrix M, diag_tag D = diag::explicit_diagonal_t>
auto diagonal(M&& m, D d = {}) {
  return diagonal_view<std::remove_cvref_t<M>, D>(std::forward<M>(m));
}

// The symmetric matrix T + S^T, where T is the `u` triangle of `m` (with
// diagonal handling `d`) and S its strict part.
template <matrix M, uplo_tag U, diag_tag D = diag::explicit_diagonal_t>
auto symmetric(M&& m, U u, D d = {}) {
  return symmetric_view<std::remove_cvref_t<M>, U, D>(std::forward<M>(m));
}

// The Hermitian matrix T + S^H, with T and S as for symmetric; only the real
// part of the stored diagonal is used (as LAPACK's ?hemv). For real value
// types this is view::symmetric.
template <matrix M, uplo_tag U, diag_tag D = diag::explicit_diagonal_t>
auto hermitian(M&& m, U u, D d = {}) {
  return hermitian_view<std::remove_cvref_t<M>, U, D>(std::forward<M>(m));
}

} // namespace view

} // namespace spblas
