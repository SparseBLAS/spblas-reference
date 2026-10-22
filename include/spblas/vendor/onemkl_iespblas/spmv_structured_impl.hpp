#pragma once

#include <algorithm>
#include <memory_resource>
#include <stdexcept>

#include <spblas/algorithms/spmv_structured_impl.hpp>
#include <spblas/concepts.hpp>
#include <spblas/detail/api_concepts.hpp>
#include <spblas/detail/log.hpp>
#include <spblas/detail/operation_state.hpp>
#include <spblas/detail/view_analysis.hpp>

#include "detail/operand.hpp"
#include "detail/structure.hpp"
#include "mkl_wrappers.hpp"
#include "spmv_impl.hpp"

//
// oneMKL IE SpMV with structure views: y = alpha * A * x [+ beta * z], via
// mkl_sparse_?_mv with a TRIANGULAR / DIAGONAL / SYMMETRIC / HERMITIAN
// matrix_descr (SpTRMV, SpDIAMV, SpSYMV, SpHEMV)
//
//   multiply_inspect([policy,] state, alpha, A, x, [beta, z,] y)
//     With a matrix_opt base: sets an mv hint for this structure and
//     optimizes its cached handle.
//   multiply([policy,] state, alpha, A, x, [beta, z,] y)
//
//   A     any structured view stack over CSR, CSC or a matrix_opt of either
//         that detail/structure.hpp maps onto MKL (all but implicit_zero,
//         which keeps the reference kernel of
//         algorithms/spmv_structured_impl.hpp).
//   x, y, z  as for spmv_impl.hpp.
//
//   These overloads are more constrained than the reference ones (the same
//   constraints and mkl_structure), so they take precedence.
//

namespace spblas {

namespace __mkl_iespblas {

template <typename A, typename X, typename Z, typename Y>
concept structured_spmv_operands =
    mkl_structure<A> && __ranges::contiguous_range<Z> &&
    __ranges::contiguous_range<Y> &&
    std::same_as<structure_scalar_t<A>,
                 std::remove_cvref_t<tensor_scalar_t<X>>> &&
    std::same_as<structure_scalar_t<A>, tensor_scalar_t<Z>> &&
    std::same_as<structure_scalar_t<A>, tensor_scalar_t<Y>>;

} // namespace __mkl_iespblas

// y = alpha * A * x + beta * z
template <__detail::scalar Alpha, matrix A, vector X, __detail::scalar Beta,
          vector Z, vector Y>
  requires(__detail::structured_view<A> && __backend::lookupable<X> &&
           __backend::lookupable<Z> && __backend::lookupable<Y> &&
           __mkl_iespblas::structured_spmv_operands<A, X, Z, Y>)
void multiply(operation_state& state, Alpha alpha, A&& a, X&& x, Beta beta,
              Z&& z, Y&& y) {
  log_trace("");
  const auto shape = __detail::analyze_view(a).shape();
  if (shape[0] != __backend::shape(y) || shape[1] != __backend::shape(x) ||
      __backend::shape(z) != __backend::shape(y)) {
    throw std::invalid_argument(
        "multiply: matrix and vector dimensions are incompatible.");
  }

  using T = __mkl_iespblas::structure_scalar_t<A>;
  const T beta_v = static_cast<T>(beta);

  auto [operand, descr] = __mkl_iespblas::structured_operand(a);
  __mkl_iespblas::acquired_handle h(operand, state.get_allocator());

  std::pmr::vector<T> x_scratch(state.get_allocator());
  const T* x_data = __mkl_iespblas::vector_input<T>(x, x_scratch);

  // MKL computes y = alpha * op(A) * x + beta * y.
  if (beta_v != T(0) && __ranges::data(z) != __ranges::data(y)) {
    std::copy(__ranges::begin(z), __ranges::end(z), __ranges::begin(y));
  }

  using J = typename decltype(h)::J;
  __mkl_iespblas::mv(h.op(), static_cast<T>(alpha), h.get(), descr, x_data,
                     J(__backend::shape(x)), beta_v, __ranges::data(y),
                     J(__backend::shape(y)), state.get_allocator());
}

template <__detail::scalar Alpha, matrix A, vector X, vector Y>
  requires(__detail::structured_view<A> && __backend::lookupable<X> &&
           __backend::lookupable<Y> &&
           __mkl_iespblas::structured_spmv_operands<A, X, Y, Y>)
void multiply_inspect(operation_state& state, Alpha alpha, A&& a, X&& x,
                      Y&& y) {
  log_trace("");
  auto [operand, descr] = __mkl_iespblas::structured_operand(a);
  if (operand.opt == nullptr) {
    return; // nowhere to keep an optimized handle
  }
  __mkl_iespblas::acquired_handle h(operand, state.get_allocator());
  if (h.cached()) {
    using J = typename decltype(h)::J;
    __mkl_iespblas::set_mv_hint(h.get(), __mkl_iespblas::mkl_operation(h.op()),
                                descr,
                                J(__mkl_iespblas::inspect_expected_calls));
    __mkl_iespblas::optimize(h.get());
  }
}

template <__detail::scalar Alpha, matrix A, vector X, __detail::scalar Beta,
          vector Z, vector Y>
  requires(__detail::structured_view<A> && __backend::lookupable<X> &&
           __backend::lookupable<Z> && __backend::lookupable<Y> &&
           __mkl_iespblas::structured_spmv_operands<A, X, Z, Y>)
void multiply_inspect(operation_state& state, Alpha alpha, A&& a, X&& x,
                      Beta beta, Z&& z, Y&& y) {
  log_trace("");
  multiply_inspect(state, alpha, std::forward<A>(a), std::forward<X>(x), y);
}

} // namespace spblas
