#pragma once

#include <memory_resource>
#include <stdexcept>

#include <spblas/algorithms/spmm_structured_impl.hpp>
#include <spblas/concepts.hpp>
#include <spblas/detail/api_concepts.hpp>
#include <spblas/detail/log.hpp>
#include <spblas/detail/operation_state.hpp>
#include <spblas/detail/view_analysis.hpp>

#include "detail/operand.hpp"
#include "detail/structure.hpp"
#include "mkl_wrappers.hpp"
#include "spmm_impl.hpp"
#include "spmv_impl.hpp"

//
// oneMKL IE SpMM with structure views: Y = alpha * A * X [+ beta * Z], via
// mkl_sparse_?_mm with a TRIANGULAR / DIAGONAL / SYMMETRIC / HERMITIAN
// matrix_descr
//
//   multiply_inspect([policy,] state, alpha, A, X, [beta, Z,] Y)
//     With a matrix_opt base: sets an mm hint for this structure and
//     optimizes its cached handle.
//   multiply([policy,] state, alpha, A, X, [beta, Z,] Y)
//
//   A        as for spmv_structured_impl.hpp (implicit_zero keeps the
//            reference kernel).
//   X, Y, Z  as for spmm_impl.hpp.
//

namespace spblas {

namespace __mkl_iespblas {

template <typename A, typename X, typename Z, typename Y>
concept structured_spmm_operands =
    mkl_structure<A> && is_plain_mdspan<Y>() &&
    std::same_as<structure_scalar_t<A>,
                 std::remove_cvref_t<tensor_scalar_t<X>>> &&
    std::same_as<structure_scalar_t<A>,
                 std::remove_cvref_t<tensor_scalar_t<Z>>> &&
    std::same_as<structure_scalar_t<A>, tensor_scalar_t<Y>>;

} // namespace __mkl_iespblas

// Y = alpha * A * X + beta * Z
template <__detail::scalar Alpha, matrix A, matrix X, __detail::scalar Beta,
          matrix Z, matrix Y>
  requires(__detail::structured_view<A> && __backend::lookupable<X> &&
           __backend::lookupable<Z> && __backend::lookupable<Y> &&
           __mkl_iespblas::structured_spmm_operands<A, X, Z, Y>)
void multiply(operation_state& state, Alpha alpha, A&& a, X&& x, Beta beta,
              Z&& z, Y&& y) {
  log_trace("");
  const auto shape = __detail::analyze_view(a).shape();
  if (shape[0] != __backend::shape(y)[0] ||
      __backend::shape(x)[1] != __backend::shape(y)[1] ||
      shape[1] != __backend::shape(x)[0] ||
      __backend::shape(z)[0] != __backend::shape(y)[0] ||
      __backend::shape(z)[1] != __backend::shape(y)[1]) {
    throw std::invalid_argument(
        "multiply: matrix dimensions are incompatible.");
  }

  using T = __mkl_iespblas::structure_scalar_t<A>;
  const T beta_v = static_cast<T>(beta);

  auto [operand, descr] = __mkl_iespblas::structured_operand(a);
  __mkl_iespblas::acquired_handle h(operand, state.get_allocator());
  using J = typename decltype(h)::J;

  auto y_d = __mkl_iespblas::describe_dense<J>(y);
  std::pmr::vector<T> x_scratch(state.get_allocator());
  auto x_d = __mkl_iespblas::dense_input<J, T>(x, y_d.layout, x_scratch);

  // MKL computes Y = alpha * op(A) * X + beta * Y.
  if (beta_v != T(0)) {
    bool aliased = false;
    if constexpr (__mkl_iespblas::direct_dense<Z>) {
      aliased = __mkl_iespblas::describe_dense<J>(z).data == y_d.data;
    }
    if (!aliased) {
      for (tensor_index_t<Y> i = 0; i < __backend::shape(y)[0]; i++) {
        for (tensor_index_t<Y> j = 0; j < __backend::shape(y)[1]; j++) {
          __backend::lookup(y, i, j) = __backend::lookup(z, i, j);
        }
      }
    }
  }

  __mkl_iespblas::mm(h.op(), static_cast<T>(alpha), h.get(), descr, y_d.layout,
                     x_d.data, J(__backend::shape(x)[0]),
                     J(__backend::shape(y)[1]), x_d.ld, beta_v, y_d.data,
                     J(__backend::shape(y)[0]), y_d.ld, state.get_allocator());
}

template <__detail::scalar Alpha, matrix A, matrix X, matrix Y>
  requires(__detail::structured_view<A> && __backend::lookupable<X> &&
           __backend::lookupable<Y> &&
           __mkl_iespblas::structured_spmm_operands<A, X, Y, Y>)
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
    __mkl_iespblas::set_mm_hint(
        h.get(), __mkl_iespblas::mkl_operation(h.op()), descr,
        __mkl_iespblas::describe_dense<J>(y).layout, J(__backend::shape(y)[1]),
        J(__mkl_iespblas::inspect_expected_calls));
    __mkl_iespblas::optimize(h.get());
  }
}

template <__detail::scalar Alpha, matrix A, matrix X, __detail::scalar Beta,
          matrix Z, matrix Y>
  requires(__detail::structured_view<A> && __backend::lookupable<X> &&
           __backend::lookupable<Z> && __backend::lookupable<Y> &&
           __mkl_iespblas::structured_spmm_operands<A, X, Z, Y>)
void multiply_inspect(operation_state& state, Alpha alpha, A&& a, X&& x,
                      Beta beta, Z&& z, Y&& y) {
  log_trace("");
  multiply_inspect(state, alpha, std::forward<A>(a), std::forward<X>(x), y);
}

} // namespace spblas
