#include <iostream>
#include <spblas/spblas.hpp>

#include <cuda_runtime.h>

#include "util.hpp"

#include <fmt/core.h>
#include <fmt/ranges.h>

int main(int argc, char** argv) {
  namespace md = spblas::__mdspan;

  using value_t = float;
  using index_t = spblas::index_t;
  using offset_t = spblas::offset_t;

  spblas::index_t m = 100;
  spblas::index_t n = 10;
  spblas::index_t k = 100;
  spblas::index_t nnz_in = 10;

  fmt::print("\n\t###########################################################"
             "######################");
  fmt::print("\n\t### Running SpMM Example:");
  fmt::print("\n\t###");
  fmt::print("\n\t###   Y = alpha * A * X");
  fmt::print("\n\t###");
  fmt::print("\n\t### with ");
  fmt::print("\n\t### A, in CSR format, of size ({}, {}) with nnz = {}", m, n,
             nnz_in);
  fmt::print("\n\t### X, a dense matrix, of size ({}, {})", n, k);
  fmt::print("\n\t### Y, a dense matrix, of size ({}, {})", m, k);
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

  CUDA_CHECK(cudaMalloc(&d_values, values.size() * sizeof(value_t)));
  CUDA_CHECK(cudaMalloc(&d_rowptr, rowptr.size() * sizeof(offset_t)));
  CUDA_CHECK(cudaMalloc(&d_colind, colind.size() * sizeof(index_t)));

  CUDA_CHECK(cudaMemcpy(d_values, values.data(),
                        values.size() * sizeof(value_t), cudaMemcpyDefault));
  CUDA_CHECK(cudaMemcpy(d_rowptr, rowptr.data(),
                        rowptr.size() * sizeof(offset_t), cudaMemcpyDefault));
  CUDA_CHECK(cudaMemcpy(d_colind, colind.data(),
                        colind.size() * sizeof(index_t), cudaMemcpyDefault));

  spblas::csr_view<value_t, index_t, offset_t> a(d_values, d_rowptr, d_colind,
                                                 shape, nnz);

  // Scale every value of `a` by 5 in place.
  // scale(5.f, a);

  std::vector<value_t> x(n * k, 1);
  std::vector<value_t> y(m * k, 0);

  value_t* d_x;
  value_t* d_y;

  CUDA_CHECK(cudaMalloc(&d_x, x.size() * sizeof(value_t)));
  CUDA_CHECK(cudaMalloc(&d_y, y.size() * sizeof(value_t)));

  CUDA_CHECK(
      cudaMemcpy(d_x, x.data(), x.size() * sizeof(value_t), cudaMemcpyDefault));
  CUDA_CHECK(
      cudaMemcpy(d_y, y.data(), y.size() * sizeof(value_t), cudaMemcpyDefault));

  md::mdspan x_span(d_x, n, k);
  md::mdspan y_span(d_y, m, k);

  // Y = A * X
  spblas::operation_info_t info;
  spblas::multiply(info, a, x_span, y_span);

  CUDA_CHECK(
      cudaMemcpy(y.data(), d_y, y.size() * sizeof(value_t), cudaMemcpyDefault));

  fmt::print("\tExample is completed!\n");

  return 0;
}
