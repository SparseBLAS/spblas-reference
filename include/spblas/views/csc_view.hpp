#pragma once

#include <spblas/views/csc_wrapper.hpp>

namespace spblas {

// Deprecated: use csc_wrapper. Kept until every backend has migrated.
template <typename T, std::integral I = index_t, std::integral O = I>
using csc_view = csc_wrapper<T, I, O>;

} // namespace spblas
