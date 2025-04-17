#pragma once

#include <spblas/backend/concepts.hpp>
#include <spblas/backend/cpos.hpp>
#include <spblas/concepts.hpp>

namespace spblas {

// Scale a tensor of type `T` by a scaling factor of type `S`.
template <typename S, typename T>
class scaled_view;

} // namespace spblas
