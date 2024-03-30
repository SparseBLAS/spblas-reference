#pragma once

#include <version>

#if defined(__cpp_lib_ranges) && __cpp_lib_ranges >= 201911L &&                \
    defined(__cpp_lib_ranges_zip) && __cpp_lib_ranges_zip >= 202110L

#include <ranges>

namespace spblas {

namespace __ranges = ::std::ranges;

namespace __detail {

namespace __ranges {

template <typename T>
concept view = ::std::ranges::view<T>;

}

} // namespace __detail

} // namespace spblas

#elif __has_include(<range/v3/all.hpp>)

#include <range/v3/all.hpp>

namespace spblas {

namespace __ranges = ::ranges;

namespace __detail {

namespace __ranges {

template <typename T>
concept view = ::ranges::view_<T>;

}

} // namespace __detail

} // namespace spblas

#else
static_assert(
    false,
    "spblas requires support for std::ranges.  Compile with C++23 or later.");

#endif
