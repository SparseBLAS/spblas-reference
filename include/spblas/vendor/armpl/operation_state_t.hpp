#pragma once

#include <spblas/vendor/armpl/detail/armpl.hpp>

namespace spblas {

namespace __mkl {

struct operation_state_t {
  armpl_spmat_t a_handle = nullptr;
  armpl_spmat_t b_handle = nullptr;
  armpl_spmat_t c_handle = nullptr;
  armpl_spmat_t d_handle = nullptr;

  operation_state_t() = default;

  operation_state_t(armpl_spmat_t a_handle, armpl_spmat_t b_handle,
                    armpl_spmat_t c_handle, armpl_spmat_t d_handle)
      : a_handle(a_handle), b_handle(b_handle), c_handle(c_handle),
        d_handle(d_handle) {}

  operation_state_t(operation_state_t&& other) {
    *this = std::move(other);
  }

  operation_state_t& operator=(operation_state_t&& other) {
    a_handle = other.a_handle;
    b_handle = other.b_handle;
    c_handle = other.c_handle;
    d_handle = other.d_handle;

    other.a_handle = other.b_handle = other.c_handle = other.d_handle = nullptr;

    return *this;
  }

  operation_state_t(const operation_state_t& other) = delete;

  ~operation_state_t() {
    release_matrix_handle(a_handle);
    release_matrix_handle(b_handle);
    release_matrix_handle(c_handle);
    release_matrix_handle(d_handle);
  }

private:
  void release_matrix_handle(armpl_spmat_t& handle) {
    if (handle != nullptr) {
      armpl_spmat_destroy(q, &handle);
    }
  }
};

} // namespace __mkl

} // namespace spblas
