#include <spblas/spblas.hpp>

#include <fmt/ranges.h>

#include <iostream>

#include "allocator.hpp"

int main(int argc, char** argv) {
  using namespace spblas;

  /**
   * 3 0 0 3
   * 0 1 0 0
   * 0 0 2 0
   * -5 0 0 4
   */
  int nnz = 6;
  std::vector<float> values{3, 3, 1, 2, -5, 4};
  std::vector<int> rowptr{0, 2, 3, 4, 6};
  std::vector<int> colind{0, 3, 1, 2, 0, 3};
  spblas::index<int> shape(4, 4);

  float* dvalues;
  int *drowptr, *dcolind;
  auto allocator = std::make_shared<const cuda_allocator>();
  spblas::spmv_handle_t spmv_handle(allocator);

  cudaMalloc((void**) &dvalues, sizeof(float) * nnz);
  cudaMalloc((void**) &drowptr, sizeof(int) * (shape[0] + 1));
  cudaMalloc((void**) &dcolind, sizeof(int) * nnz);
  cudaMemcpy(dvalues, values.data(), sizeof(float) * nnz,
             cudaMemcpyHostToDevice);
  cudaMemcpy(drowptr, rowptr.data(), sizeof(int) * (shape[0] + 1),
             cudaMemcpyHostToDevice);
  cudaMemcpy(dcolind, colind.data(), sizeof(int) * nnz, cudaMemcpyHostToDevice);
  csr_view<float, int> a(dvalues, drowptr, dcolind, shape, nnz);

  simple_operation_handle_t handle(allocator);

  scale(handle, 4.0f, a);
  cudaMemcpy(values.data(), dvalues, sizeof(float) * nnz,
             cudaMemcpyDeviceToHost);

  auto inf_norm = matrix_inf_norm(handle, a);
  auto frob_norm = matrix_frob_norm(handle, a);

  std::cout << "inf norm: " << inf_norm << ", which should be 36" << std::endl;
  std::cout << "frob norm: " << frob_norm << ", which should be 32"
            << std::endl;

  cudaFree(dvalues);
  cudaFree(drowptr);
  cudaFree(dcolind);
  return 0;
}
