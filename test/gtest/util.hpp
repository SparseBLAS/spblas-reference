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

} // namespace util
