#pragma once

#include "exception.hpp"
#include <hip/hip_runtime.h>
#include <spblas/allocator.hpp>

namespace spblas::detail {

/**
 * When user want to have the stream support for allocation. they might do it in
 * the following way.
 * ```
 * class rocm_stream_allocator: public spblas::allocator {
 * public:
 *   stream_allocator(hipStream_t stream): stream_(stream) {
 *   }
 *   void* alloc(size_t size) override {
 *     void* ptr;
 *     throw_if_error(hipMallocAsync(&ptr, size, stream_));
 *     return ptr;
 *   }
 *   void free(void* ptr) override {
 *     throw_if_error(hipFreeAsync(ptr, stream));
 *   }
 * private:
 *   hipStream_t stream_;
 * };
 *
 * TODO: we either take the stream information from the allocator when execution
 * policy is not passed, or user must use it correctly with the corresponding
 * execution policy.
 */

// This is the blocking version of memory allocation and deallocation.
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
