#pragma once

#include <version>

#if __has_include(<mdspan>)
#include <mdspan>
#endif

#if defined(__cpp_lib_mdspan) && __cpp_lib_mdspan >= 202207L

namespace spblas {
namespace __mdspan = std;
}

#elif __has_include(<experimental/mdspan>)

#include <experimental/mdspan>

namespace spblas {
namespace __mdspan = std::experimental;
}

#else

static_assert(false, "spblas requires mdspan.  Compile with a C++23 compiler "
                     "or download the std/experimental implementation.");

#endif

namespace spblas{
// Define templated aliases for col_major (layout_left) and row_major
// (layout_right) mdspan types.
template <typename I, typename T>
using mdspan_col_major = __mdspan::mdspan<
    T,
    __mdspan::extents<I, __mdspan::dynamic_extent, __mdspan::dynamic_extent>,
    __mdspan::layout_left>;

template <typename I, typename T>
using mdspan_row_major = __mdspan::mdspan<
    T,
    __mdspan::extents<I, __mdspan::dynamic_extent, __mdspan::dynamic_extent>,
    __mdspan::layout_right>;
}
