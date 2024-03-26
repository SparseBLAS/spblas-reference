#pragma once

#include <fmt/ranges.h>
#include <oneapi/mkl.hpp>

namespace spblas {

namespace __mkl {

struct operation_state_t {
  oneapi::mkl::sparse::matrix_handle_t a_handle = nullptr;
  oneapi::mkl::sparse::matrix_handle_t b_handle = nullptr;
  oneapi::mkl::sparse::matrix_handle_t c_handle = nullptr;
  oneapi::mkl::sparse::matrix_handle_t d_handle = nullptr;

  oneapi::mkl::sparse::matmat_descr_t descr;

  void* c_rowptr = nullptr;

  sycl::queue q;

  operation_state_t() = default;

  operation_state_t(oneapi::mkl::sparse::matrix_handle_t a_handle,
                    oneapi::mkl::sparse::matrix_handle_t b_handle,
                    oneapi::mkl::sparse::matrix_handle_t c_handle,
                    oneapi::mkl::sparse::matrix_handle_t d_handle,
                    oneapi::mkl::sparse::matmat_descr_t descr, void* c_rowptr,
                    sycl::queue q)
      : a_handle(a_handle), b_handle(b_handle), c_handle(c_handle),
        d_handle(d_handle), descr(descr), c_rowptr(c_rowptr), q(q) {}
  operation_state_t(operation_state_t&& other) {
    *this = std::move(other);
  }

  operation_state_t& operator=(operation_state_t&& other) {
    a_handle = other.a_handle;
    b_handle = other.b_handle;
    c_handle = other.c_handle;
    d_handle = other.d_handle;

    descr = other.descr;
    c_rowptr = other.c_rowptr;
    q = other.q;

    other.a_handle = other.b_handle = other.c_handle = other.d_handle = nullptr;
    other.descr = nullptr;
    other.c_rowptr = nullptr;

    return *this;
  }

  operation_state_t(const operation_state_t& other) = delete;

  ~operation_state_t() {
    fmt::print("Freeing handles, etc...\n");
    oneapi::mkl::sparse::release_matrix_handle(q, &a_handle);
    oneapi::mkl::sparse::release_matrix_handle(q, &b_handle);
    oneapi::mkl::sparse::release_matrix_handle(q, &c_handle);
    oneapi::mkl::sparse::release_matrix_handle(q, &d_handle);
    oneapi::mkl::sparse::release_matmat_descr(&descr);
  }
};

} // namespace __mkl

} // namespace spblas
