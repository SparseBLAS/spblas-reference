#pragma once

#include <spblas/vendor/onemkl_sycl/detail/get_pointer_device.hpp>
#include <sycl/sycl.hpp>

namespace spblas {

namespace mkl {

class parallel_policy {
public:
  parallel_policy() {}

  template <typename T>
  sycl::queue get_queue(T* ptr) const {
    return spblas::__mkl::get_pointer_queue(ptr);
  }

  sycl::queue get_queue() const {
    return sycl::queue(sycl::default_selector_v);
  }
};

class device_policy {
public:
  device_policy(const sycl::queue& queue) : queue_(queue) {}

  sycl::queue& get_queue() {
    return queue_;
  }

  const sycl::queue& get_queue() const {
    return queue_;
  }

  sycl::device get_device() const {
    return queue_.get_device();
  }

  sycl::context get_context() const {
    return queue_.get_context();
  }

private:
  sycl::queue queue_;
};

inline parallel_policy par;

} // namespace mkl

} // namespace spblas
