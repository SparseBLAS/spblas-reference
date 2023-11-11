#pragma once

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
