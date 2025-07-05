#pragma once

#include "exception.hpp"
#include <cuda_runtime.h>

namespace spblas {

namespace cusparse {

template <typename T, std::size_t Alignment = 0>
class cuda_allocator {
public:
  using value_type = T;
  using pointer = T*;
  using const_pointer = const T*;
  using reference = T&;
  using const_reference = const T&;
  using size_type = std::size_t;
  using difference_type = std::ptrdiff_t;

  cuda_allocator() noexcept {}
  cuda_allocator(cudaStream_t stream) noexcept : stream_(stream) {}

  template <typename U>
  cuda_allocator(const cuda_allocator<U, Alignment>& other) noexcept
      : stream_(other.stream()) {}

  cuda_allocator(const cuda_allocator&) = default;
  cuda_allocator& operator=(const cuda_allocator&) = default;
  ~cuda_allocator() = default;

  using is_always_equal = std::false_type;

  pointer allocate(std::size_t size) {
    void* ptr;
    this->throw_if_failure(cudaMallocAsync(&ptr, size * sizeof(T), stream()));

    return reinterpret_cast<T*>(ptr);
  }

  void deallocate(pointer ptr, std::size_t n = 0) {
    if (ptr != nullptr) {
      this->throw_if_failure(cudaFreeAsync(ptr, stream()));
    }
  }

  bool operator==(const cuda_allocator&) const = default;
  bool operator!=(const cuda_allocator&) const = default;

  template <typename U>
  struct rebind {
    using other = cuda_allocator<U, Alignment>;
  };

  cudaStream_t stream() const noexcept {
    return this->stream_;
  }

private:
  void throw_if_failure(cudaError_t error) {
    if (error != cudaSuccess) {
      throw std::bad_alloc{};
    }
  }

  cudaStream_t stream_ = nullptr;
};

} // namespace cusparse

} // namespace spblas
