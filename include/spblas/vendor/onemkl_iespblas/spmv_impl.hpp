#pragma once

#include <memory_resource>
#include <stdexcept>

#include <spblas/concepts.hpp>
#include <spblas/detail/api_concepts.hpp>
#include <spblas/detail/log.hpp>
#include <spblas/detail/operation_state.hpp>

#include "detail/operand.hpp"
#include "mkl_wrappers.hpp"

//
// oneMKL IE SpMV: y = alpha * A * x [+ beta * z], via mkl_sparse_?_mv
//
//   multiply_inspect([policy,] state, alpha, A, x, [beta, z,] y)
//     With a matrix_opt: sets an mv hint and optimizes its cached handle.
//     Otherwise there is nowhere to keep the result, so it does nothing.
//   multiply([policy,] state, alpha, A, x, [beta, z,] y)
//
//   A     CSR or CSC, a matrix_opt of either, view::transpose /
//         view::conjugate of those (see detail/operand.hpp). conj(A) without
//         a transpose is done by the wrapper, still with A's (cached) handle.
//   x     contiguous vector, or a view (e.g. view::conjugate) that is
//         copied first.
//   y, z  contiguous vectors. z may alias y; if beta == 0, z is not read.
//   All with the same value type, float / double / complex.
//

namespace spblas {

namespace __mkl_iespblas {

// Expected multiply calls passed to the hint when inspecting a matrix_opt:
// the user asked for optimization, so assume it will be reused.
inline constexpr int inspect_expected_calls = 1000;

template <typename A, typename X, typename Z, typename Y>
concept spmv_operands =
    sparse_input<A> && __backend::lookupable<X> &&
    __ranges::contiguous_range<Z> && __ranges::contiguous_range<Y> &&
    std::same_as<tensor_scalar_t<A>, std::remove_cvref_t<tensor_scalar_t<X>>> &&
    std::same_as<tensor_scalar_t<A>, tensor_scalar_t<Z>> &&
    std::same_as<tensor_scalar_t<A>, tensor_scalar_t<Y>>;

} // namespace __mkl_iespblas

// y = alpha * A * x + beta * z
template <__detail::scalar Alpha, matrix A, vector X, __detail::scalar Beta,
          vector Z, vector Y>
  requires(__mkl_iespblas::spmv_operands<A, X, Z, Y>)
void multiply(operation_state& state, Alpha alpha, A&& a, X&& x, Beta beta,
              Z&& z, Y&& y) {
  log_trace("");
  if (__backend::shape(a)[0] != __backend::shape(y) ||
      __backend::shape(a)[1] != __backend::shape(x) ||
      __backend::shape(z) != __backend::shape(y)) {
    throw std::invalid_argument(
        "multiply: matrix and vector dimensions are incompatible.");
  }

  using T = tensor_scalar_t<A>;
  const T beta_v = static_cast<T>(beta);

  auto operand = __mkl_iespblas::analyze(a);
  __mkl_iespblas::acquired_handle h(operand, state.get_allocator());

  std::pmr::vector<T> x_scratch(state.get_allocator());
  const T* x_data = __mkl_iespblas::vector_input<T>(x, x_scratch);

  // MKL computes y = alpha * op(A) * x + beta * y.
  if (beta_v != T(0) && __ranges::data(z) != __ranges::data(y)) {
    std::copy(__ranges::begin(z), __ranges::end(z), __ranges::begin(y));
  }

  using J = typename decltype(h)::J;
  __mkl_iespblas::mv(h.op(), static_cast<T>(alpha), h.get(),
                     __mkl_iespblas::general_descr(), x_data,
                     J(__backend::shape(x)), beta_v, __ranges::data(y),
                     J(__backend::shape(y)), state.get_allocator());
}

// y = alpha * A * x
template <__detail::scalar Alpha, matrix A, vector X, vector Y>
  requires(__mkl_iespblas::spmv_operands<A, X, Y, Y>)
void multiply(operation_state& state, Alpha alpha, A&& a, X&& x, Y&& y) {
  log_trace("");
  multiply(state, alpha, std::forward<A>(a), std::forward<X>(x),
           tensor_scalar_t<Y>(0), y, y);
}

template <__detail::scalar Alpha, matrix A, vector X, vector Y>
  requires(__mkl_iespblas::spmv_operands<A, X, Y, Y>)
void multiply_inspect(operation_state& state, Alpha alpha, A&& a, X&& x,
                      Y&& y) {
  log_trace("");
  auto operand = __mkl_iespblas::analyze(a);
  if (operand.opt == nullptr) {
    return; // nowhere to keep an optimized handle
  }
  __mkl_iespblas::acquired_handle h(operand, state.get_allocator());
  if (h.cached()) {
    using J = typename decltype(h)::J;
    __mkl_iespblas::set_mv_hint(h.get(), __mkl_iespblas::mkl_operation(h.op()),
                                __mkl_iespblas::general_descr(),
                                J(__mkl_iespblas::inspect_expected_calls));
    __mkl_iespblas::optimize(h.get());
  }
}

template <__detail::scalar Alpha, matrix A, vector X, __detail::scalar Beta,
          vector Z, vector Y>
  requires(__mkl_iespblas::spmv_operands<A, X, Z, Y>)
void multiply_inspect(operation_state& state, Alpha alpha, A&& a, X&& x,
                      Beta beta, Z&& z, Y&& y) {
  log_trace("");
  multiply_inspect(state, alpha, std::forward<A>(a), std::forward<X>(x), y);
}

} // namespace spblas
