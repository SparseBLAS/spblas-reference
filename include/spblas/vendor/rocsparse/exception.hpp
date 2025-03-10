#pragma once

#include <hip/hip_runtime.h>
#include <rocsparse/rocsparse.h>
#include <stdexcept>
#include <string>

namespace spblas::detail {

// throw an exception if the hipError_t is not hipSuccess.
void throw_if_error(hipError_t error_code) {
  if (error_code == hipSuccess) {
    return;
  }
  std::string name = hipGetErrorName(error_code);
  std::string message = hipGetErrorString(error_code);
  throw std::runtime_error(name + ":" + message);
}

// throw an exception if the rocsparse_status is not rocsparse_status_success.
void throw_if_error(rocsparse_status error_code) {
#define REGISTER_ROCSPARSE_ERROR(error_name)                                   \
  if (error_code == error_name) {                                              \
    throw std::runtime_error(#error_name);                                     \
  }

  if (error_code == rocsparse_status_success) {
    return;
  }

  REGISTER_ROCSPARSE_ERROR(rocsparse_status_invalid_handle);
  REGISTER_ROCSPARSE_ERROR(rocsparse_status_not_implemented);
  REGISTER_ROCSPARSE_ERROR(rocsparse_status_invalid_pointer);
  REGISTER_ROCSPARSE_ERROR(rocsparse_status_invalid_size);
  REGISTER_ROCSPARSE_ERROR(rocsparse_status_memory_error);
  REGISTER_ROCSPARSE_ERROR(rocsparse_status_internal_error);
  REGISTER_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
  REGISTER_ROCSPARSE_ERROR(rocsparse_status_arch_mismatch);
  REGISTER_ROCSPARSE_ERROR(rocsparse_status_zero_pivot);
  REGISTER_ROCSPARSE_ERROR(rocsparse_status_not_initialized);
  REGISTER_ROCSPARSE_ERROR(rocsparse_status_type_mismatch);
#undef REGISTER_ROCSPARSE_ERROR

  throw std::runtime_error("Unknown error from rocsparse_status");
}

} // namespace spblas::detail
