#pragma once

#include <cusparse.h>
#include <memory>

namespace spblas {
namespace __cusparse {

class abstract_operation_state_t {
public:
  virtual ~abstract_operation_state_t() = default;

  // Common state that all operations need
  cusparseHandle_t handle() const {
    return handle_;
  }

protected:
  abstract_operation_state_t() {
    cusparseCreate(&handle_);
  }

  virtual ~abstract_operation_state_t() {
    if (handle_) {
      cusparseDestroy(handle_);
    }
  }

  cusparseHandle_t handle_;
};

} // namespace __cusparse
} // namespace spblas
