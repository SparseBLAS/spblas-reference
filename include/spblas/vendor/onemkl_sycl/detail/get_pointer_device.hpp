#pragma once

#include <sycl/sycl.hpp>
#include <vector>

namespace spblas {

namespace __mkl {

inline std::vector<sycl::context> global_contexts_;

template <typename T>
std::pair<sycl::device, sycl::context> get_pointer_device(T* ptr) {
  if (global_contexts_.empty()) {
    for (auto&& platform : sycl::platform::get_platforms()) {
      sycl::context context(platform.get_devices());

      global_contexts_.push_back(context);
    }
  }

  for (auto&& context : global_contexts_) {
    try {
      sycl::device device = sycl::get_pointer_device(ptr, context);
      return {device, context};
    } catch (...) {
    }
  }

  throw std::runtime_error(
      "get_pointer_device: could not locate device corresponding to pointer");
}

template <typename T>
sycl::queue get_pointer_queue(T* ptr) {
  try {
    auto&& [device, context] = get_pointer_device(ptr);
    return sycl::queue(context, device);
  } catch (...) {
    return sycl::queue(sycl::cpu_selector_v);
  }
}

} // namespace __mkl

} // namespace spblas
