#pragma once

#include <cuda_runtime.h>

#define CUDA_CHECK(expression)                                                 \
  do {                                                                         \
    const cudaError_t status = expression;                                     \
    if (status != cudaSuccess) {                                               \
      std::cerr << "CUDA error " << status << ": "                             \
                << cudaGetErrorString(status) << " at " << __FILE__ << ":"     \
                << __LINE__ << std::endl;                                      \
    }                                                                          \
  } while (false)
