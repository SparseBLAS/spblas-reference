#pragma once

#include <spblas/views/views.hpp>

namespace spblas {

namespace __detail {

// Inspector for csr_view

template <typename T>
struct is_instantiation_of_csr_view {
  static constexpr bool value = false;
};

template <typename T,
          std::integral I,
          std::integral O>
struct is_instantiation_of_csr_view<csr_view<T, I, O>> {
  static constexpr bool value = true;
};

template <typename T>
static constexpr bool is_csr_view_v = is_instantiation_of_csr_view<std::remove_cvref_t<T>>::value;

} // namespace __detail

} // namespace spblas