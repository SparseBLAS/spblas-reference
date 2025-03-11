#pragma once

#include <iostream>
#include <spblas/allocator.hpp>
#include <vector>

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

template <typename T>
class device_type_allocator {
public:
  using value_type = T;

  T* allocate(std::size_t n) {
    T* ptr;
    spblas::detail::throw_if_error(hipMalloc(&ptr, n * sizeof(T)));
    return ptr;
  }

  void deallocate(T* ptr, std::size_t) {
    spblas::detail::throw_if_error(hipFree(ptr));
  }
};

#else

#include <algorithm>
#include <memory>

template <typename ValueType>
void copy_to_device(std::size_t num, const ValueType* input,
                    ValueType* output) {
  std::copy(input, input + num, output);
}

template <typename ValueType>
void copy_to_host(std::size_t num, const ValueType* input, ValueType* output) {
  std::copy(input, input + num, output);
}

template <typename T>
using device_type_allocator = std::allocator<T>;

#endif

template <typename T>
using device_vector = std::vector<T, device_type_allocator<T>>;

template <typename T>
device_vector<T> allocate_device_vector(std::size_t n) {
  device_vector<T> vector;
  // we can not use the constructor because it will try to insert the value.
  vector.reserve(n);
  return vector;
}
