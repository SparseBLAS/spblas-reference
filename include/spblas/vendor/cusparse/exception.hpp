#pragma once

#include <cuda_runtime.h>
#include <cusparse.h>
#include <stdexcept>
#include <string>

namespace spblas {

namespace __cusparse {

// Throw an exception if the cudaError_t is not cudaSuccess.
void throw_if_error(cudaError_t error_code, std::string prefix = "") {
  if (error_code == cudaSuccess) {
    return;
  }
  std::string name = cudaGetErrorName(error_code);
  std::string message = cudaGetErrorString(error_code);
  throw std::runtime_error(prefix + "CUDA encountered an error " + name +
                           ": \"" + message + "\"");
}

// Throw an exception if the cusparseStatus_t is not CUSPARSE_STATUS_SUCCESS.
void throw_if_error(cusparseStatus_t error_code) {
  if (error_code == CUSPARSE_STATUS_SUCCESS) {
    return;
  } else if (error_code == CUSPARSE_STATUS_NOT_INITIALIZED) {
    throw std::runtime_error(
        "cuSPARSE encountered an error: \"CUSPARSE_STATUS_NOT_INITIALIZED\"");
  } else if (error_code == CUSPARSE_STATUS_ALLOC_FAILED) {
    throw std::runtime_error(
        "cuSPARSE encountered an error: \"CUSPARSE_STATUS_ALLOC_FAILED\"");
  } else if (error_code == CUSPARSE_STATUS_INVALID_VALUE) {
    throw std::runtime_error(
        "cuSPARSE encountered an error: \"CUSPARSE_STATUS_INVALID_VALUE\"");
  } else if (error_code == CUSPARSE_STATUS_ARCH_MISMATCH) {
    throw std::runtime_error(
        "cuSPARSE encountered an error: \"CUSPARSE_STATUS_ARCH_MISMATCH\"");
  } else if (error_code == CUSPARSE_STATUS_EXECUTION_FAILED) {
    throw std::runtime_error(
        "cuSPARSE encountered an error: \"CUSPARSE_STATUS_EXECUTION_FAILED\"");
  } else if (error_code == CUSPARSE_STATUS_INTERNAL_ERROR) {
    throw std::runtime_error(
        "cuSPARSE encountered an error: \"CUSPARSE_STATUS_INTERNAL_ERROR\"");
  } else if (error_code == CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED) {
    throw std::runtime_error("cuSPARSE encountered an error: "
                             "\"CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED\"");
  } else if (error_code == CUSPARSE_STATUS_NOT_SUPPORTED) {
    throw std::runtime_error(
        "cuSPARSE encountered an error: \"CUSPARSE_STATUS_NOT_SUPPORTED\"");
  } else if (error_code == CUSPARSE_STATUS_INSUFFICIENT_RESOURCES) {
    throw std::runtime_error("cuSPARSE encountered an error: "
                             "\"CUSPARSE_STATUS_INSUFFICIENT_RESOURCES\"");
  } else {
    throw std::runtime_error(
        "cuSPARSE encountered an error: \"unknown error\"");
  }
}

} // namespace __cusparse

} // namespace spblas
