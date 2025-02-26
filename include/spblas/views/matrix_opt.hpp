#pragma once

#include <spblas/backend/concepts.hpp>
#include <spblas/backend/cpos.hpp>
#include <spblas/concepts.hpp>

namespace spblas {

template <matrix M>
class matrix_opt;

namespace __detail {
template <typename T>
struct is_instantiation_of_matrix_opt_view {
  static constexpr bool value = false;
};

template <typename T>
struct is_instantiation_of_matrix_opt_view<matrix_opt<T>> {
  static constexpr bool value = true;
};

template <typename T>
static constexpr bool is_matrix_opt_view_v =
    is_instantiation_of_matrix_opt_view<std::remove_cvref_t<T>>::value;

} // namespace __detail

} // namespace spblas
