#include <spblas/spblas.hpp>

#include <fmt/ranges.h>

#include <iostream>

#include "allocator.hpp"

int main(int argc, char** argv) {
  using namespace spblas;
  namespace md = spblas::__mdspan;

  // clang-format off
  std::vector<float> dense_values{
    3, 0, 0, 3,
    0, 1, 0, 0,
    0, 0, 2, 0,
   -5, 0, 0, 4};
  // clang-format on
  spblas::index<int> shape(4, 4);

  float* ddense_values;
  auto allocator = std::make_shared<const cuda_allocator>();

  cudaMalloc((void**) &ddense_values, sizeof(float) * shape[0] * shape[1]);
  cudaMemcpy(ddense_values, dense_values.data(),
             sizeof(float) * shape[0] * shape[1], cudaMemcpyHostToDevice);
  md::mdspan dense(ddense_values, shape[0], shape[1]);
  int* drowptr;
  cudaMalloc((void**) &drowptr, sizeof(int) * (shape[0] + 1));
  std::span<int> rowptr_span(drowptr, shape[0] + 1);
  csr_view<float, int> csr(nullptr, drowptr, nullptr, shape, 0);

  conversion_handle_t handle(allocator);
  conversion_compute(handle, dense, csr);
  auto nnz = handle.result_nnz();

  int* dcolind;
  float* dvalues;
  std::cout << "nnz " << nnz << std::endl;
  cudaMalloc((void**) &dvalues, sizeof(float) * nnz);

  cudaMalloc((void**) &dcolind, sizeof(int) * nnz);

  std::span<int> colind_span(dcolind, nnz);
  std::span<float> values_span(dvalues, nnz);
  csr.update(values_span, rowptr_span, colind_span, shape,
             static_cast<int>(nnz));
  conversion_execute(handle, dense, csr);
  std::vector<float> values(nnz);
  std::vector<int> rowptr(shape[0] + 1);
  std::vector<int> colind(nnz);
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
  cudaFree(ddense_values);
  cudaFree(drowptr);
  cudaFree(dcolind);
  return 0;
}
