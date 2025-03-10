#pragma once

#include <hip/hip_runtime.h>
#include <spblas/allocator.hpp>

namespace spblas::detail {

class rocm_allocator : public spblas::allocator {
public:
  void* alloc(size_t size) override {
    void* ptr;
    hipMalloc(&ptr, size);
    return ptr;
  }

  void free(void* ptr) override {
    hipFree(ptr);
  }
};

} // namespace spblas::detail
