#pragma once

#include <concepts>
#include <memory_resource>
#include <stdexcept>
#include <utility>

#include <spblas/detail/backend_state.hpp>
#include <spblas/detail/types.hpp>

namespace spblas {

// State for one SparseBLAS operation, passed to every stage of it
// (`_inspect`, `_compute`, `_fill`, ...) and reusable across repeated calls.
// Library-internal allocations use the memory resource given at construction,
// or the backend's default resource if none is given.
class operation_state {
public:
  operation_state() : operation_state(__detail::default_memory_resource()) {}

  explicit operation_state(std::pmr::memory_resource* resource)
      : resource_(resource != nullptr ? resource
                                      : __detail::default_memory_resource()) {}

  explicit operation_state(const std::pmr::polymorphic_allocator<>& allocator)
      : operation_state(allocator.resource()) {}

  operation_state(operation_state&&) = default;
  operation_state& operator=(operation_state&&) = default;
  operation_state(const operation_state&) = delete;
  operation_state& operator=(const operation_state&) = delete;

  // Number of stored entries in the sparse output, valid once the stage that
  // computes it (e.g. `multiply_compute`) has run.
  offset_t get_result_nnz() const noexcept {
    return result_nnz_;
  }

  std::pmr::polymorphic_allocator<> get_allocator() const noexcept {
    return std::pmr::polymorphic_allocator<>(resource_);
  }

  // For implementations only. `result_nnz` may use the output matrix's own
  // offset type; throws std::overflow_error if it is not representable as
  // offset_t, rather than silently truncating it.
  template <std::integral N>
  void update_impl_(N result_nnz) {
    if (!std::in_range<offset_t>(result_nnz)) {
      throw std::overflow_error(
          "spblas: result nnz is not representable as spblas::offset_t");
    }
    result_nnz_ = static_cast<offset_t>(result_nnz);
  }

  // For implementations only: backend-specific state.
  __detail::backend_operation_state_t state_;

private:
  std::pmr::memory_resource* resource_;
  offset_t result_nnz_ = 0;
};

} // namespace spblas
