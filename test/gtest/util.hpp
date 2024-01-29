#pragma once

#include <tuple>
#include <vector>

namespace util {

inline auto dims =
    std::vector({std::tuple(1000, 100, 100), std::tuple(100, 1000, 10000),
                 std::tuple(40, 40, 1000)});

}
