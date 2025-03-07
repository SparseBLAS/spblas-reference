#pragma once

#include <cstddef>

namespace spblas {

/**
 * allocator base class. When user provides the allocator implementation should
 * inherit from this class.
 */
class allocator {
public:
  virtual void alloc(void** ptrptr, std::size_t size) const = 0;

  virtual void free(void* ptr) const = 0;
};

} // namespace spblas
