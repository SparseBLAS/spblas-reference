#pragma once

#include <utility>

#include <spblas/detail/api_concepts.hpp>
#include <spblas/detail/log.hpp>
#include <spblas/detail/operation_state.hpp>

// Optional leading execution policy for the multiply and triangular-solve
// families, for backends that do not interpret policies (the reference and
// onemkl_iespblas backends): the policy is dropped and the call forwarded.
// A backend that does interpret policies provides its own, more specific
// overloads instead of including this header.

namespace spblas {

template <__detail::execution_policy P, typename... Args>
void multiply(P&& policy, operation_state& state, Args&&... args) {
  log_trace("");
  multiply(state, std::forward<Args>(args)...);
}

template <__detail::execution_policy P, typename... Args>
void multiply_inspect(P&& policy, operation_state& state, Args&&... args) {
  log_trace("");
  multiply_inspect(state, std::forward<Args>(args)...);
}

template <__detail::execution_policy P, typename... Args>
void multiply_compute(P&& policy, operation_state& state, Args&&... args) {
  log_trace("");
  multiply_compute(state, std::forward<Args>(args)...);
}

template <__detail::execution_policy P, typename... Args>
void multiply_fill(P&& policy, operation_state& state, Args&&... args) {
  log_trace("");
  multiply_fill(state, std::forward<Args>(args)...);
}

template <__detail::execution_policy P, typename... Args>
void multiply_compute_symbolic(P&& policy, operation_state& state,
                               Args&&... args) {
  log_trace("");
  multiply_compute_symbolic(state, std::forward<Args>(args)...);
}

template <__detail::execution_policy P, typename... Args>
void multiply_fill_symbolic(P&& policy, operation_state& state,
                            Args&&... args) {
  log_trace("");
  multiply_fill_symbolic(state, std::forward<Args>(args)...);
}

template <__detail::execution_policy P, typename... Args>
void multiply_numeric(P&& policy, operation_state& state, Args&&... args) {
  log_trace("");
  multiply_numeric(state, std::forward<Args>(args)...);
}

template <__detail::execution_policy P, typename... Args>
void triangular_solve(P&& policy, operation_state& state, Args&&... args) {
  log_trace("");
  triangular_solve(state, std::forward<Args>(args)...);
}

template <__detail::execution_policy P, typename... Args>
void triangular_solve_inspect(P&& policy, operation_state& state,
                              Args&&... args) {
  log_trace("");
  triangular_solve_inspect(state, std::forward<Args>(args)...);
}

} // namespace spblas
