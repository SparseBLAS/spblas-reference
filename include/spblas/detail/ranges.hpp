#pragma once

#if __has_include(<ranges>)
#include <ranges>
#endif

#if defined(__cpp_lib_ranges) && __cpp_lib_ranges >= 201911L

namespace spblas {

namespace __ranges = std::ranges;

}

#else
static_assert(false, "spblas requires support for std::ranges.  Compile with C++20 or later.");

#endif
