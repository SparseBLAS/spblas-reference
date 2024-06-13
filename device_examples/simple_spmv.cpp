#include <spblas/spblas.hpp>

#include <fmt/ranges.h>

#include <iostream>

// #include "allocator.hpp"
class cuda_allocator : public spblas::allocator {
public:
  void alloc(void** ptrptr, size_t size) const override {
    cudaMalloc(ptrptr, size);
  }

  void free(void* ptr) const override {
    cudaFree(ptr);
  }
};

int main(int argc, char** argv) {
  using namespace spblas;

  auto&& [values, rowptr, colind, shape, nnz] =
      generate_csr<float, int>(100, 100, 10);

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

  // Scale every value of `a` by 5 in place.
  // scale(5.f, a);

  std::vector<float> b(100, 1);
  std::vector<float> c(100, 0);
  float *db, *dc;
  cudaMalloc((void**) &db, sizeof(float) * 100);
  cudaMalloc((void**) &dc, sizeof(float) * 100);
  cudaMemcpy(db, b.data(), sizeof(float) * 100, cudaMemcpyHostToDevice);
  cudaMemcpy(dc, c.data(), sizeof(float) * 100, cudaMemcpyHostToDevice);

  std::span<float> b_span(db, 100);
  std::span<float> c_span(dc, 100);

  float alpha = 2.0f;
  // c = a * alpha * b
  // multiply(a, scaled(alpha, b), c);
  multiply(spmv_handle, a, b_span, c_span);

  cudaMemcpy(c.data(), dc, sizeof(float) * 100, cudaMemcpyDeviceToHost);
  for (int i = 0; i < 100; i++) {
    std::cout << c.at(i) << " ";
  }
  std::cout << std::endl;
  cudaFree(dvalues);
  cudaFree(drowptr);
  cudaFree(dcolind);
  cudaFree(db);
  cudaFree(dc);
  return 0;
}
