#include <spblas/spblas.hpp>

#include <fmt/ranges.h>

#include <iostream>

#include "allocator.hpp"

int main(int argc, char** argv) {
  using namespace spblas;

  auto&& [values, rowptr, colind, shape, nnz] =
      generate_csr<float, int>(100, 100, 10);

  // setup device csr
  float* dvalues;
  int *drowptr, *dcolind;
  cudaMalloc(&dvalues, sizeof(float) * nnz);
  cudaMalloc(&drowptr, sizeof(int) * (shape[0] + 1));
  cudaMalloc(&dcolind, sizeof(int) * nnz);
  cudaMemcpy(dvalues, values.data(), sizeof(float) * nnz,
             cudaMemcpyHostToDevice);
  cudaMemcpy(drowptr, rowptr.data(), sizeof(int) * (shape[0] + 1),
             cudaMemcpyHostToDevice);
  cudaMemcpy(dcolind, colind.data(), sizeof(int) * nnz, cudaMemcpyHostToDevice);
  csr_view<float, int> a(dvalues, drowptr, dcolind, shape, nnz);

  // setup vector
  std::vector<float> b(shape[1], 1);
  std::vector<float> c(shape[0], 0);
  float *db, *dc;
  cudaMalloc((void**) &db, sizeof(float) * shape[1]);
  cudaMalloc((void**) &dc, sizeof(float) * shape[0]);
  cudaMemcpy(db, b.data(), sizeof(float) * shape[1], cudaMemcpyHostToDevice);
  cudaMemcpy(dc, c.data(), sizeof(float) * shape[0], cudaMemcpyHostToDevice);

  std::span<float> b_span(db, shape[1]);
  std::span<float> c_span(dc, shape[0]);

  auto allocator = std::make_shared<const cuda_allocator>();
  spblas::spmv_state_t spmv_handle(allocator);
  multiply(spmv_handle, a, b_span, c_span);

  cudaMemcpy(c.data(), dc, sizeof(float) * shape[0], cudaMemcpyDeviceToHost);
  for (const auto& val : c) {
    std::cout << val << " ";
  }
  std::cout << std::endl;
  cudaFree(dvalues);
  cudaFree(drowptr);
  cudaFree(dcolind);
  cudaFree(db);
  cudaFree(dc);
  return 0;
}
