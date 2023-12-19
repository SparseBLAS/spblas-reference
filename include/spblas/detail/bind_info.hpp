#pragma once

#include <spblas/concepts.hpp>
#include <spblas/detail/operation_info_t.hpp>

namespace spblas {

template <matrix A>
struct bind_info {
  A a;
  operation_info_t info;
};

} // namespace spblas
