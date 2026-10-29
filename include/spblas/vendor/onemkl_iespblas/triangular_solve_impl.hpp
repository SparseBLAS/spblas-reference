#pragma once

#include <algorithm>
#include <memory_resource>
#include <stdexcept>

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
// oneMKL IE triangular solve: T x = alpha * b, via mkl_sparse_?_trsv
//
//   triangular_solve_inspect([policy,] state, T, [alpha,] b, x)
//     With a matrix_opt base: sets an sv hint and optimizes its cached
//     handle.
//   triangular_solve([policy,] state, T, [alpha,] b, x)
//
//   T     any composition of view::triangle / view::diagonal (also of a
//         view::symmetric / hermitian) with view::transpose / view::conjugate
//         that represents a triangular or diagonal matrix, over CSR, CSC or a
//         matrix_opt of either (__detail::triangular_structure). Mapped onto
//         a TRIANGULAR or DIAGONAL matrix_descr by detail/structure.hpp.
//         view::diag::implicit_zero is a static_assert (singular).
//   b     contiguous vector or a view of one (copied first); x contiguous.
//         x may alias b (b is then copied).
//

namespace spblas {

namespace __mkl_iespblas {

template <typename Tv, typename B, typename X>
concept trsv_operands =
    __detail::triangular_structure<Tv> && mkl_structure<Tv> &&
    __backend::lookupable<B> && __ranges::contiguous_range<X> &&
    std::same_as<structure_scalar_t<Tv>,
                 std::remove_cvref_t<tensor_scalar_t<B>>> &&
    std::same_as<structure_scalar_t<Tv>, tensor_scalar_t<X>>;

} // namespace __mkl_iespblas

template <typename Tv, __detail::scalar Alpha, vector B, vector X>
  requires(__mkl_iespblas::trsv_operands<Tv, B, X>)
void triangular_solve(operation_state& state, Tv&& t, Alpha alpha, B&& b,
                      X&& x) {
  log_trace("");
  const auto shape = __detail::analyze_view(t).shape();
  if (shape[0] != shape[1] || shape[0] != __backend::shape(b) ||
      shape[1] != __backend::shape(x)) {
    throw std::invalid_argument(
        "triangular_solve: matrix and vector dimensions are incompatible.");
  }

  using T = tensor_scalar_t<X>;
  auto [operand, descr] = __mkl_iespblas::structured_operand(t);
  __mkl_iespblas::acquired_handle h(operand, state.get_allocator().host());

  std::pmr::vector<T> b_scratch(state.get_allocator().host());
  const T* b_data = __mkl_iespblas::vector_input<T>(b, b_scratch);
  if (b_data == __ranges::data(x)) {
    // MKL's input and output must not overlap.
    b_scratch.assign(b_data, b_data + __backend::shape(x));
    b_data = b_scratch.data();
  }

  using J = typename decltype(h)::J;
  __mkl_iespblas::trsv(h.op(), static_cast<T>(alpha), h.get(), descr, b_data,
                       __ranges::data(x), J(__backend::shape(x)),
                       state.get_allocator().host());
}

// An implied zero diagonal makes the matrix singular.
template <typename Tv, __detail::scalar Alpha, vector B, vector X>
  requires(__detail::is_implicit_zero_structure_<Tv>())
void triangular_solve(operation_state& state, Tv&& t, Alpha alpha, B&& b,
                      X&& x) {
  static_assert(!__detail::is_implicit_zero_structure_<Tv>(),
                "triangular_solve: view::diag::implicit_zero is not supported "
                "(the matrix would be singular).");
}

template <typename Tv, vector B, vector X>
  requires(__mkl_iespblas::trsv_operands<Tv, B, X> ||
           __detail::is_implicit_zero_structure_<Tv>())
void triangular_solve(operation_state& state, Tv&& t, B&& b, X&& x) {
  log_trace("");
  triangular_solve(state, std::forward<Tv>(t), tensor_scalar_t<X>(1),
                   std::forward<B>(b), std::forward<X>(x));
}

template <typename Tv, __detail::scalar Alpha, vector B, vector X>
  requires(__mkl_iespblas::trsv_operands<Tv, B, X>)
void triangular_solve_inspect(operation_state& state, Tv&& t, Alpha alpha,
                              B&& b, X&& x) {
  log_trace("");
  auto [operand, descr] = __mkl_iespblas::structured_operand(t);
  if (operand.opt == nullptr) {
    return; // nowhere to keep an optimized handle
  }
  __mkl_iespblas::acquired_handle h(operand, state.get_allocator().host());
  if (h.cached()) {
    using J = typename decltype(h)::J;
    __mkl_iespblas::set_sv_hint(h.get(), __mkl_iespblas::mkl_operation(h.op()),
                                descr,
                                J(__mkl_iespblas::inspect_expected_calls));
    __mkl_iespblas::optimize(h.get());
  }
}

template <typename Tv, vector B, vector X>
  requires(__mkl_iespblas::trsv_operands<Tv, B, X>)
void triangular_solve_inspect(operation_state& state, Tv&& t, B&& b, X&& x) {
  log_trace("");
  triangular_solve_inspect(state, std::forward<Tv>(t), tensor_scalar_t<X>(1),
                           std::forward<B>(b), std::forward<X>(x));
}

} // namespace spblas
