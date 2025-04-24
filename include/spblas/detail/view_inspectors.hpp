#pragma once

#include <optional>
#include <utility> // std::declval

#include <spblas/detail/concepts.hpp>
#include <spblas/views/inspectors.hpp>

namespace spblas {

namespace __detail {

// Does this tensor view have a base?
template <typename T>
concept has_base = view<T> && requires(T& t) {
  { t.base() } -> tensor;
};

// Inspect a tensor: does it have a scaling factor?  If so, compute it.
// Returns an empty optional if no scaling factor OR returns an optional
// with the product of all the scaling factors.
template <tensor T>
auto get_scaling_factor(T&& t) {
  if constexpr (has_base<T>) {
    auto base_scaling_factor = get_scaling_factor(t.base());

    if constexpr (is_scaled_view_v<T>) {
      auto scaling_factor = t.alpha();

      using scaling_factor_type =
          decltype(scaling_factor * base_scaling_factor.value());

      if (base_scaling_factor.has_value()) {
        return std::optional<scaling_factor_type>(scaling_factor *
                                                  base_scaling_factor.value());
      } else {
        return std::optional<scaling_factor_type>(scaling_factor);
      }
    } else {
      return base_scaling_factor;
    }
  } else {
    if constexpr (is_scaled_view_v<T>) {
      return std::optional(t.alpha());
    } else {
      return std::optional<tensor_scalar_t<T>>{};
    }
  }
}

// Get scaling factors of t and u, returning:
// 1) empty optional, if no scaling factor in either
// 2) scaling factor of t OR u, if only one has a scaling factor
// 3) product of scaling factor of t and u, if both have a scaling factor.
template <tensor T, tensor U>
auto get_scaling_factor(T&& t, U&& u) {
  auto t_scaling_factor = get_scaling_factor(t);
  auto u_scaling_factor = get_scaling_factor(u);

  using scalar_type = decltype(std::declval<typename std::remove_cvref_t<
                                   decltype(t_scaling_factor)>::value_type>() *
                               std::declval<typename std::remove_cvref_t<
                                   decltype(u_scaling_factor)>::value_type>());

  if (t_scaling_factor.has_value()) {
    if (u_scaling_factor.has_value()) {
      return std::optional<scalar_type>(t_scaling_factor.value() *
                                        u_scaling_factor.value());
    } else {
      return std::optional<scalar_type>(t_scaling_factor);
    }
  } else if (u_scaling_factor.has_value()) {
    return std::optional<scalar_type>(u_scaling_factor);
  } else {
    return std::optional<scalar_type>{};
  }
}

template <tensor T>
bool has_scaling_factor(T&& t) {
  return get_scaling_factor(t).has_value();
}

template <tensor T>
auto get_ultimate_base(T&& t) {
  if constexpr (has_base<T>) {
    return get_ultimate_base(t.base());
  } else {
    return t;
  }
}

template <tensor T>
bool has_matrix_opt(T&& t) {
  if constexpr (is_matrix_opt_v<T>) {
    return true;
  } else if constexpr (has_base<T>) {
    return has_matrix_opt(t.base());
  } else {
    return false;
  }
}

template <typename T>
using ultimate_base_type_t = decltype(get_ultimate_base(std::declval<T>()));

template <typename T>
concept has_csr_base = is_csr_view_v<ultimate_base_type_t<T>>;

template <typename T>
concept has_csc_base = is_csc_view_v<ultimate_base_type_t<T>>;

template <typename T>
concept has_mdspan_matrix_base = is_matrix_mdspan_v<ultimate_base_type_t<T>>;

template <typename T>
concept has_contiguous_range_base =
    spblas::__ranges::contiguous_range<ultimate_base_type_t<T>>;

} // namespace __detail

} // namespace spblas
