#pragma once

#include <hip/hip_runtime.h>

#define HIP_CHECK(expression)                                                  \
  do {                                                                         \
    const hipError_t status = expression;                                      \
    if (status != hipSuccess) {                                                \
      std::cerr << "HIP error " << status << ": " << hipGetErrorString(status) \
                << " at " << __FILE__ << ":" << __LINE__ << std::endl;         \
    }                                                                          \
  } while (false)
