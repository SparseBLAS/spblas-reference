#pragma once

#include <concepts>
#include <type_traits>

#include <spblas/concepts.hpp>
#include <spblas/detail/operation_state.hpp>
#include <spblas/detail/type_traits.hpp>

namespace spblas {

namespace __detail {

// A scaling-factor argument (`alpha`, `beta`).
template <typename T>
concept scalar =
    std::is_arithmetic_v<std::remove_cvref_t<T>> || is_std_complex_v<T>;

template <typename T>
concept is_operation_state =
    std::same_as<std::remove_cvref_t<T>, operation_state>;

// The optional leading execution-policy argument: anything that is not the
// operation state, a tensor, or a scalar.
template <typename P>
concept execution_policy = !is_operation_state<P> && !tensor<P> && !scalar<P>;

} // namespace __detail

} // namespace spblas
