#pragma once

#include <spblas/detail/tag_invoke.hpp>

namespace spblas {

namespace __backend {

struct shape_fn_ {
  template <typename T>
    requires(spblas::is_tag_invocable_v<shape_fn_, T>)
  constexpr auto operator()(T&& t) const {
    return spblas::tag_invoke(shape_fn_{}, std::forward<T>(t));
  }
};

inline constexpr auto shape = shape_fn_{};

struct values_fn_ {
  template <typename T>
    requires(spblas::is_tag_invocable_v<values_fn_, T>)
  constexpr auto operator()(T&& t) const {
    return spblas::tag_invoke(values_fn_{}, std::forward<T>(t));
  }
};

inline constexpr auto values = values_fn_{};

struct rows_fn_ {
  template <typename T>
    requires(spblas::is_tag_invocable_v<rows_fn_, T>)
  constexpr auto operator()(T&& t) const {
    return spblas::tag_invoke(rows_fn_{}, std::forward<T>(t));
  }
};

inline constexpr auto rows = rows_fn_{};

} // namespace __backend

} // namespace spblas