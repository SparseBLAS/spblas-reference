#pragma once

#include <oneapi/mkl.hpp>
#include <sycl/sycl.hpp>

namespace spblas {

namespace __mkl {

// oneMKL SYCL optimization data held by a matrix_opt.
struct matrix_opt_state_t {
  oneapi::mkl::sparse::matrix_handle_t matrix_handle = nullptr;

  matrix_opt_state_t() = default;
  matrix_opt_state_t(const matrix_opt_state_t&) = delete;
  matrix_opt_state_t& operator=(const matrix_opt_state_t&) = delete;

  ~matrix_opt_state_t() {
    invalidate();
  }

  // Called when the library changes the matrix's data.
  void invalidate() {
    if (matrix_handle) {
      // q here needs to be on same context as queue in operations,
      // idealy from execution policy
      sycl::queue q(sycl::cpu_selector_v);
      oneapi::mkl::sparse::release_matrix_handle(q, &matrix_handle, {}).wait();
      matrix_handle = nullptr;
    }
  }
};

} // namespace __mkl

} // namespace spblas
