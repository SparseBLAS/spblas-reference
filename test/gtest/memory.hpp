#pragma once

#include <functional>
#include <memory>

template <typename T>
using device_memory_manager = std::unique_ptr<T, std::function<void(T*)>>;

#if defined(SPBLAS_ENABLE_ROCSPARSE)

#include <hip/hip_runtime.h>
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
device_memory_manager<T> allocate_device_ptr(std::size_t n) {
  T* ptr;
  spblas::detail::throw_if_error(hipMalloc(&ptr, n * sizeof(T)));
  return device_memory_manager<T>(
      ptr, [](T* ptr) { spblas::detail::throw_if_error(hipFree(ptr)); });
}

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

template <typename T>
device_memory_manager<T> allocate_device_ptr(std::size_t n) {
  T* ptr = static_cast<T*>(::operator new(n * sizeof(T), std::nothrow_t{}));
  return device_memory_manager<T>(
      ptr, [](T* ptr) { ::operator delete(ptr, std::nothrow_t{}); });
}

#endif
