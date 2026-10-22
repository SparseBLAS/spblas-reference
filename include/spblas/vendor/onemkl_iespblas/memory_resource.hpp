#pragma once

#include <algorithm>
#include <cstddef>
#include <memory_resource>
#include <new>

#include "mkl.h"

namespace spblas {

namespace __mkl_iespblas {

// Default memory resource for the IE Sparse BLAS backend: mkl_malloc/mkl_free
// with at least 64-byte alignment.
class mkl_memory_resource final : public std::pmr::memory_resource {
  static constexpr std::size_t min_alignment = 64;

  void* do_allocate(std::size_t bytes, std::size_t alignment) override {
    void* p = mkl_malloc(std::max<std::size_t>(bytes, 1),
                         static_cast<int>(std::max(alignment, min_alignment)));
    if (p == nullptr) {
      throw std::bad_alloc();
    }
    return p;
  }

  void do_deallocate(void* p, std::size_t, std::size_t) override {
    mkl_free(p);
  }

  // Any mkl_memory_resource can free memory allocated by another.
  bool
  do_is_equal(const std::pmr::memory_resource& other) const noexcept override {
    return dynamic_cast<const mkl_memory_resource*>(&other) != nullptr;
  }
};

inline std::pmr::memory_resource* default_memory_resource() noexcept {
  static mkl_memory_resource resource;
  return &resource;
}

} // namespace __mkl_iespblas

} // namespace spblas
