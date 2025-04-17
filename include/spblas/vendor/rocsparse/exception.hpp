#pragma once

#include <hip/hip_runtime.h>
#include <rocsparse/rocsparse.h>
#include <stdexcept>
#include <string>

namespace spblas {

namespace __rocsparse {

// Throw an exception if the hipError_t is not hipSuccess.
void throw_if_error(hipError_t error_code, std::string prefix = "") {
  if (error_code == hipSuccess) {
    return;
  }
  std::string name = hipGetErrorName(error_code);
  std::string message = hipGetErrorString(error_code);
  throw std::runtime_error(prefix + "HIP encountered an error " + name +
                           ": \"" + message + "\"");
}

// Throw an exception if the rocsparse_status is not rocsparse_status_success.
void throw_if_error(rocsparse_status error_code) {
  if (error_code == rocsparse_status_success) {
    return;
  } else if (error_code == rocsparse_status_invalid_handle) {
    throw std::runtime_error(
        "rocSPARSE encountered an error: \"rocsparse_status_invalid_handle\"");
  } else if (error_code == rocsparse_status_not_implemented) {
    throw std::runtime_error(
        "rocSPARSE encountered an error: \"rocsparse_status_not_implemented\"");
  } else if (error_code == rocsparse_status_invalid_pointer) {
    throw std::runtime_error(
        "rocSPARSE encountered an error: \"rocsparse_status_invalid_pointer\"");
  } else if (error_code == rocsparse_status_invalid_size) {
    throw std::runtime_error(
        "rocSPARSE encountered an error: \"rocsparse_status_invalid_size\"");
  } else if (error_code == rocsparse_status_memory_error) {
    throw std::runtime_error(
        "rocSPARSE encountered an error: \"rocsparse_status_memory_error\"");
  } else if (error_code == rocsparse_status_internal_error) {
    throw std::runtime_error(
        "rocSPARSE encountered an error: \"rocsparse_status_internal_error\"");
  } else if (error_code == rocsparse_status_invalid_value) {
    throw std::runtime_error(
        "rocSPARSE encountered an error: \"rocsparse_status_invalid_value\"");
  } else if (error_code == rocsparse_status_arch_mismatch) {
    throw std::runtime_error(
        "rocSPARSE encountered an error: \"rocsparse_status_arch_mismatch\"");
  } else if (error_code == rocsparse_status_zero_pivot) {
    throw std::runtime_error(
        "rocSPARSE encountered an error: \"rocsparse_status_zero_pivot\"");
  } else if (error_code == rocsparse_status_not_initialized) {
    throw std::runtime_error(
        "rocSPARSE encountered an error: \"rocsparse_status_not_initialized\"");
  } else if (error_code == rocsparse_status_type_mismatch) {
    throw std::runtime_error(
        "rocSPARSE encountered an error: \"rocsparse_status_type_mismatch\"");
  } else if (error_code == rocsparse_status_type_mismatch) {
    throw std::runtime_error(
        "rocSPARSE encountered an error: \"rocsparse_status_invalid_size\"");
  } else if (error_code == rocsparse_status_invalid_size) {
    throw std::runtime_error(
        "rocSPARSE encountered an error: \"rocsparse_status_invalid_size\"");
  } else if (error_code == rocsparse_status_invalid_size) {
    throw std::runtime_error(
        "rocSPARSE encountered an error: \"rocsparse_status_invalid_size\"");
  } else if (error_code == rocsparse_status_invalid_size) {
    throw std::runtime_error(
        "rocSPARSE encountered an error: \"rocsparse_status_invalid_size\"");
  } else if (error_code == rocsparse_status_invalid_size) {
    throw std::runtime_error(
        "rocSPARSE encountered an error: \"rocsparse_status_invalid_size\"");
  } else {
    throw std::runtime_error(
        "rocSPARSE encountered an error: \"unknown error\"");
  }
}

} // namespace __rocsparse

} // namespace spblas
