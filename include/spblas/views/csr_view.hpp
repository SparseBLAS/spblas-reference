#pragma once

#include <spblas/views/csr_wrapper.hpp>

namespace spblas {

// Deprecated: use csr_wrapper. Kept until every backend has migrated.
template <typename T, std::integral I = index_t, std::integral O = I>
using csr_view = csr_wrapper<T, I, O>;

} // namespace spblas
