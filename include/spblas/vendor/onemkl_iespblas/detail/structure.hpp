#pragma once

#include <type_traits>

#include <spblas/detail/view_analysis.hpp>

#include "../mkl_wrappers.hpp"
#include "operand.hpp"

//
// Structure views (view::triangle / diagonal / symmetric / hermitian, in any
// composition with transpose / conjugate) mapped onto oneMKL
//
//   __detail::analyze_view gives M = op(structure(region(base))), with region
//   R in base's coordinates. base's operand (analyze, operand.hpp) is a
//   handle H over base's CSR arrays, or over the CSR arrays of base^T for a
//   CSC base (t below). matrix_descr's fill mode refers to H, so it is R,
//   flipped when t; the MKL operation follows from op and t:
//
//     general, R lower / upper   TRIANGULAR   op(R(H^t)) = op^(T xor t)(R'(H))
//     general, R diagonal        DIAGONAL     transpose irrelevant
//     symmetric                  SYMMETRIC    S(H^T) = S(H) (flipped
//                                             triangle): t absorbed
//     hermitian                  HERMITIAN    H(B^T) = conj(H(B)) (flipped
//                                             triangle): t is a conjugate
//
//   diag: SPARSE_DIAG_UNIT for implicit_unit, else NON_UNIT.
//   implicit_zero has no MKL equivalent: those cases are not mkl_structure,
//   so multiplication keeps the reference kernel and the triangular solve
//   rejects them (singular).
//
//   MKL's HERMITIAN kernels use only the real part of the stored diagonal,
//   as view::hermitian does (checked with oneMKL 2026.0).
//

namespace spblas {

namespace __mkl_iespblas {

template <typename V>
constexpr bool is_mkl_structure_() {
  if constexpr (__detail::structured_view<V>) {
    return sparse_input<__detail::view_base_t<V>> &&
           __detail::describe_view<V>().diag !=
               __detail::diag_kind::implicit_zero;
  } else {
    return false;
  }
}

// A structured view stack over an MKL-compatible matrix that oneMKL can
// compute with.
template <typename V>
concept mkl_structure = is_mkl_structure_<V>();

// The value type of a structured view stack (its base's).
template <typename V>
using structure_scalar_t = tensor_scalar_t<__detail::view_base_t<V>>;

// The operand (handle flags) and descriptor for a structured view stack.
template <typename W>
struct structured_operand_t {
  sparse_operand<W> operand;
  matrix_descr descr;
};

template <mkl_structure V>
auto structured_operand(const V& v) {
  using __detail::diag_kind;
  using __detail::view_region;
  using __detail::view_structure;

  auto a = __detail::analyze_view(v);
  using A = decltype(a);
  auto operand = analyze(a.base);
  const bool t = operand.transpose; // H is the handle of base^T

  if constexpr (A::structure == view_structure::general) {
    operand.transpose = t != A::transpose;
    operand.conjugate = operand.conjugate != A::conjugate;
  } else if constexpr (A::structure == view_structure::symmetric) {
    operand.transpose = false;
    operand.conjugate = operand.conjugate != A::conjugate;
  } else {
    operand.transpose = false;
    operand.conjugate = (operand.conjugate != A::conjugate) != t;
  }

  matrix_descr descr;
  if constexpr (A::structure == view_structure::symmetric) {
    descr.type = SPARSE_MATRIX_TYPE_SYMMETRIC;
  } else if constexpr (A::structure == view_structure::hermitian) {
    descr.type = SPARSE_MATRIX_TYPE_HERMITIAN;
  } else if constexpr (A::region == view_region::diagonal) {
    descr.type = SPARSE_MATRIX_TYPE_DIAGONAL;
  } else {
    descr.type = SPARSE_MATRIX_TYPE_TRIANGULAR;
  }
  const view_region r = t ? __detail::transposed_region(A::region) : A::region;
  descr.mode =
      r == view_region::upper ? SPARSE_FILL_MODE_UPPER : SPARSE_FILL_MODE_LOWER;
  descr.diag = A::diag == diag_kind::implicit_unit ? SPARSE_DIAG_UNIT
                                                   : SPARSE_DIAG_NON_UNIT;
  return structured_operand_t<decltype(operand.csr)>{operand, descr};
}

} // namespace __mkl_iespblas

} // namespace spblas
