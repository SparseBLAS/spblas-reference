#pragma once

#include <memory>
#include <rocsparse/rocsparse.h>

namespace spblas {
namespace __rocsparse {

class abstract_operation_state_t {
public:
  // Common state that all operations need
  rocsparse_handle handle() const {
    return handle_;
  }

  // Make std::default_delete a friend so unique_ptr can delete us
  friend struct std::default_delete<abstract_operation_state_t>;

protected:
  abstract_operation_state_t() {
    rocsparse_create_handle(&handle_);
  }

  virtual ~abstract_operation_state_t() {
    if (handle_) {
      rocsparse_destroy_handle(handle_);
    }
  }

  rocsparse_handle handle_;
};

} // namespace __rocsparse
} // namespace spblas
