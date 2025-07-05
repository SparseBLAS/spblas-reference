#pragma once

#include <spblas/vendor/onemkl_sycl/detail/execution_policy.hpp>

namespace spblas {

namespace __mkl {

template <typename T>
sycl::queue get_queue(const spblas::mkl::parallel_policy& policy, T* ptr) {
  return policy.get_queue(ptr);
}

template <typename T>
sycl::queue& get_queue(spblas::mkl::device_policy& policy, T* ptr) {
  return policy.get_queue();
}

} // namespace __mkl

} // namespace spblas

#if __has_include(<thrust/execution_policy.h>)

#include <thrust/execution_policy.h>

namespace spblas {

namespace __mkl {

template <typename T>
sycl::queue& get_queue(thrust::execution_policy& policy, T* ptr) {
  return policy.get_queue();
}

} // namespace __mkl

} // namespace spblas

#endif
