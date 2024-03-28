#pragma once

#include <spblas/backend/concepts.hpp>
#include <spblas/backend/cpos.hpp>
#include <spblas/concepts.hpp>

namespace spblas {

// Scale a tensor of type `T` by a scaling factor of type `S`.
template <typename S, typename T>
class scaled_view;

namespace __detail {

template <typename T>
struct is_instantiation_of_scaled_view {
  static constexpr bool value = false;
};

template <typename S, typename T>
struct is_instantiation_of_scaled_view<scaled_view<S, T>> {
  static constexpr bool value = true;
};

template <typename T>
static constexpr bool is_scaled_view_v =
    is_instantiation_of_scaled_view<std::remove_cvref_t<T>>::value;

template <typename T>
static constexpr bool is_scaled_view_matrix_v =
    is_scaled_view_v<T> && matrix<decltype(std::declval<T>().base())>;

} // namespace __detail

} // namespace spblas
