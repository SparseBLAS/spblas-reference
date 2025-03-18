#include <iostream>
#include <spblas/spblas.hpp>

#include <hip/hip_runtime.h>

#include "util.hpp"

#include <fmt/core.h>
#include <fmt/ranges.h>

int main(int argc, char** argv) {
  using namespace spblas;

  using T = float;
  using I = spblas::index_t;

  spblas::index_t m = 100;
  spblas::index_t n = 100;
  spblas::index_t nnz_in = 10;

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
      generate_csr<T, spblas::index_t>(m, n, nnz_in);

  T* d_values;
  I* d_rowptr;
  I* d_colind;

  HIP_CHECK(hipMalloc(&d_values, values.size() * sizeof(T)));
  HIP_CHECK(hipMalloc(&d_rowptr, rowptr.size() * sizeof(I)));
  HIP_CHECK(hipMalloc(&d_colind, colind.size() * sizeof(I)));

  HIP_CHECK(hipMemcpy(d_values, values.data(), values.size() * sizeof(T),
                      hipMemcpyDefault));
  HIP_CHECK(hipMemcpy(d_rowptr, rowptr.data(), rowptr.size() * sizeof(I),
                      hipMemcpyDefault));
  HIP_CHECK(hipMemcpy(d_colind, colind.data(), colind.size() * sizeof(I),
                      hipMemcpyDefault));

  csr_view<T, I> a(d_values, d_rowptr, d_colind, shape, nnz);

  // Scale every value of `a` by 5 in place.
  // scale(5.f, a);

  std::vector<T> x(n, 1);
  std::vector<T> y(m, 0);

  T* d_x;
  T* d_y;

  HIP_CHECK(hipMalloc(&d_x, x.size() * sizeof(T)));
  HIP_CHECK(hipMalloc(&d_y, y.size() * sizeof(T)));

  HIP_CHECK(hipMemcpy(d_x, x.data(), x.size() * sizeof(T), hipMemcpyDefault));
  HIP_CHECK(hipMemcpy(d_y, y.data(), y.size() * sizeof(T), hipMemcpyDefault));

  std::span<T> x_span(d_x, n);
  std::span<T> y_span(d_y, m);

  // y = A * x
  multiply(a, x_span, y_span);

  HIP_CHECK(hipMemcpy(y.data(), d_y, y.size() * sizeof(T), hipMemcpyDefault));

  fmt::print("\tExample is completed!\n");

  return 0;
}
