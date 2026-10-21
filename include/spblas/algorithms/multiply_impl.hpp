#pragma once

#include <utility>

#include <spblas/detail/api_concepts.hpp>
#include <spblas/detail/operation_state.hpp>

//
// Reference multiply, one file per operation:
//
//   SpMV    spmv_impl.hpp    sparse A, dense vectors x, y [, z]
//   SpMM    spmm_impl.hpp    sparse A, dense matrices X, Y [, Z]
//   SpGEMM  spgemm_impl.hpp  sparse A, B, C (kernels in detail/spgemm/)
//
// Each file lists the formats and views it supports. Variants for structured
// views of A (view::triangle / symmetric / hermitian) belong in their own
// files next to these, e.g. spmv_triangle_impl.hpp.
//

#include <spblas/algorithms/spgemm_impl.hpp>
#include <spblas/algorithms/spmm_impl.hpp>
#include <spblas/algorithms/spmv_impl.hpp>

namespace spblas {

// Optional leading execution policy, for every multiply variant. The
// reference implementation runs sequentially and ignores it.
template <__detail::execution_policy P, typename... Args>
void multiply(P&& policy, operation_state& state, Args&&... args) {
  multiply(state, std::forward<Args>(args)...);
}

template <__detail::execution_policy P, typename... Args>
void multiply_inspect(P&& policy, operation_state& state, Args&&... args) {
  multiply_inspect(state, std::forward<Args>(args)...);
}

template <__detail::execution_policy P, typename... Args>
void multiply_compute(P&& policy, operation_state& state, Args&&... args) {
  multiply_compute(state, std::forward<Args>(args)...);
}

template <__detail::execution_policy P, typename... Args>
void multiply_fill(P&& policy, operation_state& state, Args&&... args) {
  multiply_fill(state, std::forward<Args>(args)...);
}

template <__detail::execution_policy P, typename... Args>
void multiply_compute_symbolic(P&& policy, operation_state& state,
                               Args&&... args) {
  multiply_compute_symbolic(state, std::forward<Args>(args)...);
}

template <__detail::execution_policy P, typename... Args>
void multiply_fill_symbolic(P&& policy, operation_state& state,
                            Args&&... args) {
  multiply_fill_symbolic(state, std::forward<Args>(args)...);
}

template <__detail::execution_policy P, typename... Args>
void multiply_numeric(P&& policy, operation_state& state, Args&&... args) {
  multiply_numeric(state, std::forward<Args>(args)...);
}

} // namespace spblas
