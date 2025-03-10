#pragma once

#include <spblas/allocator.hpp>

#if defined(SPBLAS_ENABLE_ROCSPARSE)

#include <hip/hip_runtime.h>
#include <spblas/vendor/rocsparse/allocator.hpp>
#include <spblas/vendor/rocsparse/exception.hpp>

template <typename ValueType>
void copy_to_device(std::size_t num, const ValueType* input,
                    ValueType* output) {
  spblas::detail::throw_if_error(
      hipMemcpy(output, input, num * sizeof(ValueType), hipMemcpyHostToDevice));
}

template <typename ValueType>
void copy_to_host(std::size_t num, const ValueType* input, ValueType* output) {
  spblas::detail::throw_if_error(
      hipMemcpy(output, input, num * sizeof(ValueType), hipMemcpyDeviceToHost));
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
  void* alloc(std::size_t size) override {
    void* ptr = ::operator new(size, std::nothrow_t{});
    return ptr;
  };

  void free(void* ptr) override {
    ::operator delete(ptr, std::nothrow_t{});
  }
};
#endif
