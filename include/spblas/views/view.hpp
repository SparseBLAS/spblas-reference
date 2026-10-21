#pragma once

#include <type_traits>
#include <utility>

#include <spblas/algorithms/conjugated_impl.hpp>
#include <spblas/algorithms/transposed.hpp>
#include <spblas/concepts.hpp>
#include <spblas/views/triangle_view.hpp>

// View functions from the SparseBLAS paper (`spblas::view::...`). Views are
// lightweight and non-owning; implementations may still materialize them
// (e.g. an explicit transpose inside a matrix_opt).
//
// Not yet provided: view::diagonal, view::symmetric, view::hermitian.

namespace spblas {

namespace view {

// Transposed view of `m`. A CSR matrix is exposed as CSC and vice versa.
template <matrix M>
auto transpose(M&& m) {
  return transposed(std::forward<M>(m));
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
