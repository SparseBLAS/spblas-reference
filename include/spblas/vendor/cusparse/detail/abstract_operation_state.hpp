#pragma once

#include <cusparse.h>
#include <memory>

namespace spblas {
namespace __cusparse {

class abstract_operation_state_t {
public:
  // Common state that all operations need
  cusparseHandle_t handle() const {
    return handle_;
  }

  // Make std::default_delete a friend so unique_ptr can delete us
  friend struct std::default_delete<abstract_operation_state_t>;

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
