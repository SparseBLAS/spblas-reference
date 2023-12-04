#pragma once

#if __has_include(<ranges>)
#include <ranges>
#endif

#if defined(__cpp_lib_ranges) && __cpp_lib_ranges >= 201911L &&                \
    defined(__cpp_lib_ranges_zip) && __cpp_lib_ranges_zip >= 202110L

namespace spblas {

namespace __ranges = std::ranges;

}

#else
static_assert(
    false,
    "spblas requires support for std::ranges.  Compile with C++23 or later.");

#endif
