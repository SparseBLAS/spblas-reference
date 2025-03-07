#pragma once

#if defined(SPBLAS_ENABLE_ROCSPARSE)

#include <hip/hip_runtime.h>

template <typename ValueType>
void copy_to_device(std::size_t num, const ValueType* input,
                    ValueType* output) {
  hipMemcpy(output, input, num * sizeof(ValueType), hipMemcpyHostToDevice);
}

template <typename ValueType>
void copy_to_host(std::size_t num, const ValueType* input, ValueType* output) {
  hipMemcpy(output, input, num * sizeof(ValueType), hipMemcpyDeviceToHost);
}

using default_allocator = spblas::detail::rocm_allocator;

#else

#include <algorithm>

template <typename ValueType>
void copy_to_device(std::size_t num, const ValueType* input,
                    ValueType* output) {
  std::copy(input, input + num, output);
}

template <typename ValueType>
void copy_to_host(std::size_t num, const ValueType* input, ValueType* output) {
  std::copy(input, input + num, output);
}

class default_allocator : public spblas::allocator {
  void* alloc(std::size_t size) const override {
    void* ptr = ::operator new(size, std::nothrow_t{});
    return ptr;
  };

  void free(void* ptr) const override {
    ::operator delete(ptr, std::nothrow_t{});
  }
};
#endif
