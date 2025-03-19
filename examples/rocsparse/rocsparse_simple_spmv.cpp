#include <iostream>
#include <spblas/spblas.hpp>

#include <hip/hip_runtime.h>

#include "util.hpp"

#include <fmt/core.h>
#include <fmt/ranges.h>

int main(int argc, char** argv) {
  using value_t = float;
  using index_t = spblas::index_t;
  using offset_t = spblas::offset_t;

  index_t m = 100;
  index_t n = 100;
  index_t nnz_in = 10;

  fmt::print("\n\t###########################################################"
             "######################");
  fmt::print("\n\t### Running SpMV Example:");
  fmt::print("\n\t###");
  fmt::print("\n\t###   y = alpha * A * x");
  fmt::print("\n\t###");
  fmt::print("\n\t### with ");
  fmt::print("\n\t### A, in CSR format, of size ({}, {}) with nnz = {}", m, n,
             nnz_in);
  fmt::print("\n\t### x, a dense vector, of size ({}, {})", n, 1);
  fmt::print("\n\t### y, a dense vector, of size ({}, {})", m, 1);
  fmt::print("\n\t### using float and spblas::index_t (size = {} bytes)",
             sizeof(spblas::index_t));
  fmt::print("\n\t###########################################################"
             "######################");
  fmt::print("\n");

  auto&& [values, rowptr, colind, shape, nnz] =
      spblas::generate_csr<value_t, index_t, offset_t>(m, n, nnz_in);

  value_t* d_values;
  offset_t* d_rowptr;
  index_t* d_colind;

  HIP_CHECK(hipMalloc(&d_values, values.size() * sizeof(value_t)));
  HIP_CHECK(hipMalloc(&d_rowptr, rowptr.size() * sizeof(offset_t)));
  HIP_CHECK(hipMalloc(&d_colind, colind.size() * sizeof(index_t)));

  HIP_CHECK(hipMemcpy(d_values, values.data(), values.size() * sizeof(value_t),
                      hipMemcpyDefault));
  HIP_CHECK(hipMemcpy(d_rowptr, rowptr.data(), rowptr.size() * sizeof(offset_t),
                      hipMemcpyDefault));
  HIP_CHECK(hipMemcpy(d_colind, colind.data(), colind.size() * sizeof(index_t),
                      hipMemcpyDefault));

  spblas::csr_view<value_t, index_t, offset_t> a(d_values, d_rowptr, d_colind,
                                                 shape, nnz);

  // Scale every value of `a` by 5 in place.
  // scale(5.f, a);

  std::vector<value_t> x(n, 1);
  std::vector<value_t> y(m, 0);

  value_t* d_x;
  value_t* d_y;

  HIP_CHECK(hipMalloc(&d_x, x.size() * sizeof(value_t)));
  HIP_CHECK(hipMalloc(&d_y, y.size() * sizeof(value_t)));

  HIP_CHECK(
      hipMemcpy(d_x, x.data(), x.size() * sizeof(value_t), hipMemcpyDefault));
  HIP_CHECK(
      hipMemcpy(d_y, y.data(), y.size() * sizeof(value_t), hipMemcpyDefault));

  std::span<value_t> x_span(d_x, n);
  std::span<value_t> y_span(d_y, m);

  // y = A * x
  spblas::spmv_state_t state;
  spblas::multiply(state, a, x_span, y_span);

  HIP_CHECK(
      hipMemcpy(y.data(), d_y, y.size() * sizeof(value_t), hipMemcpyDefault));

  fmt::print("\tExample is completed!\n");

  return 0;
}
