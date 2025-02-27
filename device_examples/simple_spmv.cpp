#include <spblas/spblas.hpp>

#include <fmt/ranges.h>

#include <iostream>

// #include "allocator.hpp"
class amd_allocator : public spblas::allocator {
public:
  void alloc(void** ptrptr, size_t size) const override {
    hipMalloc(ptrptr, size);
  }

  void free(void* ptr) const override {
    hipFree(ptr);
  }
};

int main(int argc, char** argv) {
  using namespace spblas;

  auto&& [values, rowptr, colind, shape, nnz] =
      generate_csr<float, int>(100, 100, 10);

  float* dvalues;
  int *drowptr, *dcolind;
  auto allocator = std::make_shared<const amd_allocator>();
  spblas::spmv_handle_t spmv_handle(allocator);

  hipMalloc((void**) &dvalues, sizeof(float) * nnz);
  hipMalloc((void**) &drowptr, sizeof(int) * (shape[0] + 1));
  hipMalloc((void**) &dcolind, sizeof(int) * nnz);
  hipMemcpy(dvalues, values.data(), sizeof(float) * nnz,
             hipMemcpyHostToDevice);
  hipMemcpy(drowptr, rowptr.data(), sizeof(int) * (shape[0] + 1),
             hipMemcpyHostToDevice);
  hipMemcpy(dcolind, colind.data(), sizeof(int) * nnz, hipMemcpyHostToDevice);
  csr_view<float, int> a(dvalues, drowptr, dcolind, shape, nnz);

  // Scale every value of `a` by 5 in place.
  // scale(5.f, a);

  std::vector<float> b(100, 1);
  std::vector<float> c(100, 0);
  float *db, *dc;
  hipMalloc((void**) &db, sizeof(float) * 100);
  hipMalloc((void**) &dc, sizeof(float) * 100);
  hipMemcpy(db, b.data(), sizeof(float) * 100, hipMemcpyHostToDevice);
  hipMemcpy(dc, c.data(), sizeof(float) * 100, hipMemcpyHostToDevice);

  std::span<float> b_span(db, 100);
  std::span<float> c_span(dc, 100);

  float alpha = 2.0f;
  // c = a * alpha * b
  // multiply(a, scaled(alpha, b), c);
  multiply(spmv_handle, a, b_span, c_span);

  hipMemcpy(c.data(), dc, sizeof(float) * 100, hipMemcpyDeviceToHost);
  for (int i = 0; i < 100; i++) {
    std::cout << c.at(i) << " ";
  }
  std::cout << std::endl;
  hipFree(dvalues);
  hipFree(drowptr);
  hipFree(dcolind);
  hipFree(db);
  hipFree(dc);
  return 0;
}
