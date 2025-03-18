#include <iostream>
#include <spblas/spblas.hpp>

#include <thrust/device_vector.h>

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

  thrust::device_vector<T> d_values(values);
  thrust::device_vector<I> d_rowptr(rowptr);
  thrust::device_vector<I> d_colind(colind);

  csr_view<T, I> a(d_values.data().get(), d_rowptr.data().get(),
                   d_colind.data().get(), shape, nnz);

  // Scale every value of `a` by 5 in place.
  // scale(5.f, a);

  std::vector<T> x(n, 1);
  std::vector<T> y(m, 0);

  thrust::device_vector<T> d_x(x);
  thrust::device_vector<T> d_y(y);

  std::span<T> x_span(d_x.data().get(), n);
  std::span<T> y_span(d_y.data().get(), m);

  // y = A * x
  multiply(a, x_span, y_span);

  thrust::copy(d_y.begin(), d_y.end(), y.begin());

  fmt::print("\tExample is completed!\n");

  return 0;
}
