#pragma once

#include <gtest/gtest.h>
#include <tuple>
#include <vector>

#define EXPECT_EQ_(t, u)                                                       \
  if constexpr (std::floating_point<std::remove_cvref_t<decltype((t))>> ||     \
                std::floating_point<std::remove_cvref_t<decltype((u))>>) {     \
    auto epsilon =                                                             \
        64 *                                                                   \
        std::numeric_limits<std::remove_cvref_t<decltype((t))>>::epsilon();    \
    auto abs_th =                                                              \
        std::numeric_limits<std::remove_cvref_t<decltype((t))>>::min();        \
    auto diff = std::abs((t) - (u));                                           \
    auto norm = std::min(                                                      \
        std::abs((t)) + std::abs((u)),                                         \
        std::numeric_limits<std::remove_cvref_t<decltype((t))>>::max());       \
    auto abs_error = std::max(abs_th, epsilon * norm);                         \
    EXPECT_NEAR((t), (u), abs_error);                                          \
  } else {                                                                     \
    EXPECT_EQ((t), (u));                                                       \
  }

namespace util {

inline auto dims =
    std::vector({std::tuple(1000, 100, 100), std::tuple(100, 1000, 10000),
                 std::tuple(40, 40, 1000)});

/*
template <typename T, typename U>
inline void expect_eq(T&& t, U&& u) {
  if constexpr(std::floating_point<T> || std::floating_point<U>) {
    auto epsilon =
  }
    template <std::floating_point Tp>
bool is_equal(Tp a, Tp b,
              Tp epsilon = 128 * std::numeric_limits<Tp>::epsilon()) {
  if (a == b) {
    return true;
  }
  auto abs_th = std::numeric_limits<Tp>::min();
  auto diff = std::abs(a - b);
  auto norm =
      std::min(std::abs(a) + std::abs(b), std::numeric_limits<Tp>::max());

  return diff < std::max(abs_th, epsilon * norm);
}
  if constexpr(std::is_same_v<std::remove_cvref_t<T>, double> ||
std::is_same_v<std::remove_cvref_t<U>, double>) { EXPECT_DOUBLE_EQ(t, u); } else
if constexpr(std::is_same_v<std::remove_cvref_t<T>, float> ||
std::is_same_v<std::remove_cvref_t<U>, float>) { EXPECT_FLOAT_EQ(t, u); } else {
    EXPECT_EQ(t, u);
  }
}
*/

} // namespace util
