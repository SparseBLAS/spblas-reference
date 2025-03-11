#pragma once

#include <cstddef>

namespace spblas {

/**
 * allocator base class. When user provides the allocator implementation should
 * inherit from this class.
 */
class allocator {
public:
  virtual void* alloc(std::size_t size) = 0;

  virtual void free(void* ptr) = 0;
};

} // namespace spblas
