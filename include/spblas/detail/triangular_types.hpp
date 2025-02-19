#pragma once

namespace spblas {

struct upper_triangle_t {
  explicit upper_triangle_t() = default;
};
inline constexpr upper_triangle_t upper_triangle{};

struct lower_triangle_t {
  explicit lower_triangle_t() = default;
};
inline constexpr lower_triangle_t lower_triangle{};

struct implicit_unit_diagonal_t {
  explicit implicit_unit_diagonal_t() = default;
};
inline constexpr implicit_unit_diagonal_t implicit_unit_diagonal{};

struct explicit_diagonal_t {
  explicit explicit_diagonal_t() = default;
};
inline constexpr explicit_diagonal_t explicit_diagonal{};

} // namespace spblas
