/*
 * Copyright (c) 2025      Advanced Micro Devices, Inc. All Rights reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */

#pragma once

#include "aoclsparse.h"

namespace spblas {

namespace __aoclsparse {

struct operation_state_t {
  aoclsparse_matrix a_handle = nullptr;
  aoclsparse_matrix b_handle = nullptr;
  aoclsparse_matrix c_handle = nullptr;

  operation_state_t() = default;

  operation_state_t(aoclsparse_matrix a_handle, aoclsparse_matrix b_handle,
                    aoclsparse_matrix c_handle)
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
  void release_matrix_handle(aoclsparse_matrix handle) {
    if (handle != nullptr) {
      aoclsparse_destroy(&handle);
    }
  }
};

} // namespace __aoclsparse

} // namespace spblas
