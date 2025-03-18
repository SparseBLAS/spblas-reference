#pragma once

#include "exception.hpp"
#include <hip/hip_runtime.h>

namespace spblas {

namespace rocsparse {

template <typename T, std::size_t Alignment = 0>
class hip_allocator {
public:
  using value_type = T;
  using pointer = T*;
  using const_pointer = const T*;
  using reference = T&;
  using const_reference = const T&;
  using size_type = std::size_t;
  using difference_type = std::ptrdiff_t;

  hip_allocator() noexcept {}
  hip_allocator(hipStream_t stream) noexcept : stream_(stream) {}

  template <typename U>
  hip_allocator(const hip_allocator<U, Alignment>& other) noexcept
      : stream_(other.stream()) {}

  hip_allocator(const hip_allocator&) = default;
  hip_allocator& operator=(const hip_allocator&) = default;
  ~hip_allocator() = default;

  using is_always_equal = std::false_type;

  pointer allocate(std::size_t size) {
    void* ptr;
    hipError_t error = hipMallocAsync(&ptr, size * sizeof(T), stream());
    throw_if_failure(error);

    return reinterpret_cast<T*>(ptr);
  }

  void deallocate(pointer ptr, std::size_t n = 0) {
    if (ptr != nullptr) {
      hipError_t error = hipFreeAsync(ptr, stream());
      throw_if_failure(error);
    }
  }

  bool operator==(const hip_allocator&) const = default;
  bool operator!=(const hip_allocator&) const = default;

  template <typename U>
  struct rebind {
    using other = hip_allocator<U, Alignment>;
  };

  hipStream_t stream() const noexcept {
    return stream_;
  }

private:
  void throw_if_failure(hipError_t error) {
    if (error != hipSuccess) {
      throw std::bad_alloc{};
    }
  }

  hipStream_t stream_ = nullptr;
};

} // namespace rocsparse

} // namespace spblas
