#pragma once

#include "operation_state_t.hpp"

namespace spblas {

namespace __mkl_iespblas {

// oneMKL data held by a matrix_opt: a handle over its CSR arrays (for a CSC
// matrix: the CSR arrays of its transpose), created on first use and
// optimized by the _inspect stages.
struct matrix_opt_state_t {
  handle_variant handle;

  // Called when the library changes the matrix's data: the handle (and any
  // optimized copy MKL keeps inside it) is dropped and rebuilt on next use.
  void invalidate() noexcept {
    handle = std::monostate{};
  }
};

} // namespace __mkl_iespblas

} // namespace spblas
