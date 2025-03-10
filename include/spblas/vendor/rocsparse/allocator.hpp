#pragma once

#include "exception.hpp"
#include <hip/hip_runtime.h>
#include <spblas/allocator.hpp>

namespace spblas::detail {

class rocm_allocator : public spblas::allocator {
public:
  void* alloc(size_t size) override {
    void* ptr;
    throw_if_error(hipMalloc(&ptr, size));
    return ptr;
  }

  void free(void* ptr) override {
    throw_if_error(hipFree(ptr));
  }
};

} // namespace spblas::detail
