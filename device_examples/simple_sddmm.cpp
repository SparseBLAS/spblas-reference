#include <spblas/spblas.hpp>

#include <fmt/ranges.h>

#include <iostream>

#include "allocator.hpp"

int main(int argc, char** argv) {
  using namespace spblas;
  namespace md = spblas::__mdspan;

  // clang-format off
  std::vector<float> a_dense_values{
    3, 0, 0, 3,
    0, 1, 0, 0,
    0, 0, 2, 0,
   -5, 0, 0, 4};
  std::vector<float> b_dense_values{
    3, 0, 3, 0,
    0, 1, 0, 0,
    0, 0, 2, 0,
    0, 0, -5, 4};
  // clang-format on
  spblas::index<int> shape(4, 4);

  float* da_dense_values;
  float* db_dense_values;
  auto allocator = std::make_shared<const cuda_allocator>();

  cudaMalloc((void**) &da_dense_values, sizeof(float) * shape[0] * shape[1]);
  cudaMemcpy(da_dense_values, a_dense_values.data(),
             sizeof(float) * shape[0] * shape[1], cudaMemcpyHostToDevice);
  md::mdspan a_dense(da_dense_values, shape[0], shape[1]);
  cudaMalloc((void**) &db_dense_values, sizeof(float) * shape[0] * shape[1]);
  cudaMemcpy(db_dense_values, b_dense_values.data(),
             sizeof(float) * shape[0] * shape[1], cudaMemcpyHostToDevice);
  md::mdspan b_dense(db_dense_values, shape[0], shape[1]);

  int nnz = 6;
  std::vector<int> rowptr{0, 2, 3, 4, 6};
  std::vector<int> colind{0, 2, 3, 0, 1, 3};
  std::vector<float> values(nnz);
  int* drowptr;
  int* dcolind;
  float* dvalues;
  cudaMalloc((void**) &drowptr, sizeof(int) * (shape[0] + 1));
  cudaMalloc((void**) &dcolind, sizeof(int) * nnz);
  cudaMalloc((void**) &dvalues, sizeof(float) * nnz);
  cudaMemcpy(drowptr, rowptr.data(), sizeof(int) * (shape[0] + 1),
             cudaMemcpyHostToDevice);
  cudaMemcpy(dcolind, colind.data(), sizeof(int) * nnz, cudaMemcpyHostToDevice);
  cudaMemcpy(dvalues, values.data(), sizeof(float) * nnz,
             cudaMemcpyHostToDevice);
  csr_view<float, int> csr(dvalues, drowptr, dcolind, shape, nnz);

  sampled_multiply_handle_t handle(allocator);
  sampled_multiply_compute(handle, a_dense, b_dense, csr);
  sampled_multiply_execute(handle, a_dense, b_dense, csr);

  cudaMemcpy(values.data(), dvalues, sizeof(float) * nnz,
             cudaMemcpyDeviceToHost);
  cudaMemcpy(colind.data(), dcolind, sizeof(int) * nnz, cudaMemcpyDeviceToHost);
  cudaMemcpy(rowptr.data(), drowptr, sizeof(int) * (shape[0] + 1),
             cudaMemcpyDeviceToHost);

  for (auto& x : rowptr) {
    std::cout << x << " ";
  }
  std::cout << std::endl;
  for (auto& x : colind) {
    std::cout << x << " ";
  }
  std::cout << std::endl;
  for (auto& x : values) {
    std::cout << x << " ";
  }
  std::cout << std::endl;

  cudaFree(dvalues);
  cudaFree(da_dense_values);
  cudaFree(db_dense_values);
  cudaFree(drowptr);
  cudaFree(dcolind);
  return 0;
}
