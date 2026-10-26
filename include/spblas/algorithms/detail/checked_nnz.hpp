#pragma once

#include <concepts>
#include <cstddef>
#include <stdexcept>
#include <utility>

namespace spblas {

namespace __detail {

// The number of entries of a sparse result, counted in std::size_t, as the
// output's offset type O. Throws std::overflow_error if it does not fit,
// rather than wrapping before operation_state can check it.
template <std::integral O>
O checked_nnz(std::size_t nnz) {
  if (!std::in_range<O>(nnz)) {
    throw std::overflow_error(
        "spblas: result nnz is not representable as the output's offset "
        "type");
  }
  return static_cast<O>(nnz);
}

} // namespace __detail

} // namespace spblas
