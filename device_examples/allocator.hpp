#pragma once
#include <spblas/spblas.hpp>

class cuda_allocator : public spblas::allocator {
public:
  void alloc(void** ptrptr, size_t size) const override {
    cudaMalloc(ptrptr, size);
  }

  void free(void* ptr) const override {
    cudaFree(ptr);
  }
};
