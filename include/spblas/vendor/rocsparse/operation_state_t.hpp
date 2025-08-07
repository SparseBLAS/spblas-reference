#pragma once

#include "detail/abstract_operation_state.hpp"
#include <memory>

namespace spblas {
namespace __rocsparse {

class operation_state_t {
public:
  operation_state_t() = default;
  operation_state_t(std::unique_ptr<abstract_operation_state_t>&& state)
      : state_(std::move(state)) {}

  // Move-only
  operation_state_t(operation_state_t&&) = default;
  operation_state_t& operator=(operation_state_t&&) = default;

  // No copying
  operation_state_t(const operation_state_t&) = delete;
  operation_state_t& operator=(const operation_state_t&) = delete;

  // Access the underlying state
  template <typename T>
  T* get_state() {
    return dynamic_cast<T*>(state_.get());
  }

  template <typename T>
  const T* get_state() const {
    return dynamic_cast<const T*>(state_.get());
  }

private:
  std::unique_ptr<abstract_operation_state_t> state_;
};

} // namespace __rocsparse
} // namespace spblas
