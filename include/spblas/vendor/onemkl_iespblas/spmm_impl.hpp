#pragma once

#include <memory_resource>
#include <stdexcept>

#include <spblas/concepts.hpp>
#include <spblas/detail/api_concepts.hpp>
#include <spblas/detail/log.hpp>
#include <spblas/detail/operation_state.hpp>

#include "detail/operand.hpp"
#include "mkl_wrappers.hpp"
#include "spmv_impl.hpp"

//
// oneMKL IE SpMM: Y = alpha * A * X [+ beta * Z], via mkl_sparse_?_mm
//
//   multiply_inspect([policy,] state, alpha, A, X, [beta, Z,] Y)
//     With a matrix_opt: sets an mm hint and optimizes its cached handle.
//   multiply([policy,] state, alpha, A, X, [beta, Z,] Y)
//
//   A     as for SpMV (detail/operand.hpp).
//   Y     row- or column-major mdspan.
//   X     used in place if it is an mdspan or view::transpose of one with
//         Y's layout; otherwise (another layout, view::conjugate, ...)
//         copied into Y's layout first.
//   Z     any dense matrix of Y's shape; may alias Y; not read if
//         beta == 0.
//   All with the same value type, float / double / complex.
//

namespace spblas {

namespace __mkl_iespblas {

template <typename A, typename X, typename Z, typename Y>
concept spmm_operands =
    sparse_input<A> && __backend::lookupable<X> && __backend::lookupable<Z> &&
    is_plain_mdspan<Y>() &&
    std::same_as<tensor_scalar_t<A>, std::remove_cvref_t<tensor_scalar_t<X>>> &&
    std::same_as<tensor_scalar_t<A>, std::remove_cvref_t<tensor_scalar_t<Z>>> &&
    std::same_as<tensor_scalar_t<A>, tensor_scalar_t<Y>>;

} // namespace __mkl_iespblas

// Y = alpha * A * X + beta * Z
template <__detail::scalar Alpha, matrix A, matrix X, __detail::scalar Beta,
          matrix Z, matrix Y>
  requires(__mkl_iespblas::spmm_operands<A, X, Z, Y>)
void multiply(operation_state& state, Alpha alpha, A&& a, X&& x, Beta beta,
              Z&& z, Y&& y) {
  log_trace("");
  if (__backend::shape(a)[0] != __backend::shape(y)[0] ||
      __backend::shape(x)[1] != __backend::shape(y)[1] ||
      __backend::shape(a)[1] != __backend::shape(x)[0] ||
      __backend::shape(z)[0] != __backend::shape(y)[0] ||
      __backend::shape(z)[1] != __backend::shape(y)[1]) {
    throw std::invalid_argument(
        "multiply: matrix dimensions are incompatible.");
  }

  using T = tensor_scalar_t<A>;
  const T beta_v = static_cast<T>(beta);

  auto operand = __mkl_iespblas::analyze(a);
  __mkl_iespblas::acquired_handle h(operand, state.get_allocator().host());
  using J = typename decltype(h)::J;

  auto y_d = __mkl_iespblas::describe_dense<J>(y);
  std::pmr::vector<T> x_scratch(state.get_allocator().host());
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

  __mkl_iespblas::mm(h.op(), static_cast<T>(alpha), h.get(),
                     __mkl_iespblas::general_descr(), y_d.layout, x_d.data,
                     J(__backend::shape(x)[0]), J(__backend::shape(y)[1]),
                     x_d.ld, beta_v, y_d.data, J(__backend::shape(y)[0]),
                     y_d.ld, state.get_allocator().host());
}

// Y = alpha * A * X
template <__detail::scalar Alpha, matrix A, matrix X, matrix Y>
  requires(__mkl_iespblas::spmm_operands<A, X, Y, Y>)
void multiply(operation_state& state, Alpha alpha, A&& a, X&& x, Y&& y) {
  log_trace("");
  multiply(state, alpha, std::forward<A>(a), std::forward<X>(x),
           tensor_scalar_t<Y>(0), y, y);
}

template <__detail::scalar Alpha, matrix A, matrix X, matrix Y>
  requires(__mkl_iespblas::spmm_operands<A, X, Y, Y>)
void multiply_inspect(operation_state& state, Alpha alpha, A&& a, X&& x,
                      Y&& y) {
  log_trace("");
  auto operand = __mkl_iespblas::analyze(a);
  if (operand.opt == nullptr) {
    return; // nowhere to keep an optimized handle
  }
  __mkl_iespblas::acquired_handle h(operand, state.get_allocator().host());
  if (h.cached()) {
    using J = typename decltype(h)::J;
    __mkl_iespblas::set_mm_hint(h.get(), __mkl_iespblas::mkl_operation(h.op()),
                                __mkl_iespblas::general_descr(),
                                __mkl_iespblas::describe_dense<J>(y).layout,
                                J(__backend::shape(y)[1]),
                                J(__mkl_iespblas::inspect_expected_calls));
    __mkl_iespblas::optimize(h.get());
  }
}

template <__detail::scalar Alpha, matrix A, matrix X, __detail::scalar Beta,
          matrix Z, matrix Y>
  requires(__mkl_iespblas::spmm_operands<A, X, Z, Y>)
void multiply_inspect(operation_state& state, Alpha alpha, A&& a, X&& x,
                      Beta beta, Z&& z, Y&& y) {
  log_trace("");
  multiply_inspect(state, alpha, std::forward<A>(a), std::forward<X>(x), y);
}

} // namespace spblas
