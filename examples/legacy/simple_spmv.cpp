#include <iostream>
#include <spblas/spblas.hpp>

#include <fmt/core.h>
#include <fmt/ranges.h>

int main(int argc, char** argv) {
  using namespace spblas;

  using T = float;

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

  csr_view<T, spblas::index_t> a(values, rowptr, colind, shape, nnz);

  // Scale every value of `a` by 5 in place.
  // scale(5.f, a);

  std::vector<T> x(n, 1);
  std::vector<T> y(m, 0);

  T alpha = 1.2f;
  auto a_scaled = scaled(alpha, a);

  // y = alpha * A * x
  multiply(a_scaled, x, y);

  fmt::print("\tExample is completed!\n");

  return 0;
}
