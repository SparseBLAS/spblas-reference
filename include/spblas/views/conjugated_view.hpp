#pragma once

#include <spblas/backend/concepts.hpp>
#include <spblas/backend/cpos.hpp>
#include <spblas/concepts.hpp>

namespace spblas {

// Conjugate a tensor of type `T`.
template <typename T>
class conjugated_view;

} // namespace spblas
