#pragma once

#include "mkl.h"

namespace spblas {

namespace __mkl_iespblas{

struct operation_state_t {
  sparse_matrix_t a_handle = nullptr;
  sparse_matrix_t b_handle = nullptr;
  sparse_matrix_t c_handle = nullptr;

  operation_state_t() = default;

  operation_state_t(sparse_matrix_t a_handle,
                    sparse_matrix_t b_handle,
                    sparse_matrix_t c_handle)
      : a_handle(a_handle), b_handle(b_handle), c_handle(c_handle) {}

  operation_state_t(operation_state_t&& other) {
    *this = std::move(other);
  }

  operation_state_t& operator=(operation_state_t&& other) {
    a_handle = other.a_handle;
    b_handle = other.b_handle;
    c_handle = other.c_handle;

    other.a_handle = other.b_handle = other.c_handle = nullptr;

    return *this;
  }

  operation_state_t(const operation_state_t& other) = delete;

  ~operation_state_t() {
    release_matrix_handle(a_handle);
    release_matrix_handle(b_handle);
    release_matrix_handle(c_handle);
  }

private:
  void release_matrix_handle(sparse_matrix_t handle) {
    if (handle != nullptr) {
      mkl_sparse_destroy(handle);
    }
  }
};

} // namespace __mkl_iespblas

} // namespace spblas
