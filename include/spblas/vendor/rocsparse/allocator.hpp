#pragma once

#include <hip/hip_runtime.h>
#include <spblas/allocator.hpp>

namespace spblas::detail {

class rocm_allocator : public spblas::allocator {
public:
  void* alloc(size_t size) const override {
    void* ptr;
    hipMalloc(&ptr, size);
    return ptr;
  }

  void free(void* ptr) const override {
    hipFree(ptr);
  }
};

} // namespace spblas::detail
