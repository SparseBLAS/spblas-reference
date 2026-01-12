#pragma once

#include <spblas/concepts.hpp>

namespace spblas {

template <matrix M>
auto conjugated(M&& m);

template <vector V>
auto conjugated(V&& v);

} // namespace spblas
