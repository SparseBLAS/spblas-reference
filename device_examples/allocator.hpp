#pragma once

#include <spblas/vendor/cusparse/multiply_impl.hpp>

class cuda_allocator : public spblas::allocator {
public:
  // we can also put the stream into consturctor to use cudaMallocAsync ...
  void alloc(void** ptrptr, size_t size) const override {
    cudaMalloc(ptrptr, size);
  }

  void free(void* ptr) const override {
    cudaFree(ptr);
  }
};
