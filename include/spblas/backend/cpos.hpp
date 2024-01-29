#pragma once

#include <spblas/detail/tag_invoke.hpp>

namespace spblas {

namespace __backend {

struct size_fn_ {
  template <typename T>
    requires(spblas::is_tag_invocable_v<size_fn_, T>)
  constexpr auto operator()(T&& t) const {
    return spblas::tag_invoke(size_fn_{}, std::forward<T>(t));
  }
};

inline constexpr auto size = size_fn_{};

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

struct lookup_fn_ {
  template <typename T, typename... Args>
    requires(spblas::is_tag_invocable_v<lookup_fn_, T, Args...>)
  constexpr tag_invoke_result_t<lookup_fn_, T, Args...>
  operator()(T&& t, Args&&... args) const {
    return spblas::tag_invoke(lookup_fn_{}, std::forward<T>(t),
                              std::forward<Args>(args)...);
  }
};

inline constexpr auto lookup = lookup_fn_{};

struct lookup_row_fn_ {
  template <typename T, typename... Args>
    requires(spblas::is_tag_invocable_v<lookup_row_fn_, T, Args...>)
  constexpr tag_invoke_result_t<lookup_row_fn_, T, Args...>
  operator()(T&& t, Args&&... args) const {
    return spblas::tag_invoke(lookup_row_fn_{}, std::forward<T>(t),
                              std::forward<Args>(args)...);
  }
};

inline constexpr auto lookup_row = lookup_row_fn_{};

} // namespace __backend

} // namespace spblas
