#pragma once

#include <spblas/detail/index.hpp>
#include <spblas/detail/types.hpp>

#ifdef SPBLAS_ENABLE_ONEMKL_SYCL
#include <spblas/vendor/onemkl_sycl/operation_state_t.hpp>
#endif

#ifdef SPBLAS_ENABLE_ARMPL
#include <spblas/vendor/armpl/operation_state_t.hpp>
#endif

#ifdef SPBLAS_ENABLE_AOCLSPARSE
#include <spblas/vendor/aoclsparse/operation_state_t.hpp>
#endif

#ifdef SPBLAS_ENABLE_CUSPARSE
#include <spblas/vendor/cusparse/operation_state_t.hpp>
#endif

namespace spblas {

class operation_info_t {
public:
  auto result_shape() {
    return result_shape_;
  }

  auto result_nnz() {
    return result_nnz_;
  }

  operation_info_t() = default;

  operation_info_t(index<> result_shape, offset_t result_nnz)
      : result_shape_(result_shape), result_nnz_(result_nnz) {}

#ifdef SPBLAS_ENABLE_ONEMKL_SYCL
  operation_info_t(index<> result_shape, offset_t result_nnz,
                   __mkl::operation_state_t&& state)
      : result_shape_(result_shape), result_nnz_(result_nnz),
        state_(std::move(state)) {}
#endif

#ifdef SPBLAS_ENABLE_ARMPL
  operation_info_t(index<> result_shape, offset_t result_nnz,
                   __armpl::operation_state_t&& state)
      : result_shape_(result_shape), result_nnz_(result_nnz),
        state_(std::move(state)) {}
#endif

#ifdef SPBLAS_ENABLE_AOCLSPARSE
  operation_info_t(index<> result_shape, offset_t result_nnz,
                   __aoclsparse::operation_state_t&& state)
      : result_shape_(result_shape), result_nnz_(result_nnz),
        state_(std::move(state)) {}
#endif

#ifdef SPBLAS_ENABLE_CUSPARSE
  operation_info_t(index<> result_shape, offset_t result_nnz,
                   __cusparse::operation_state_t&& state)
      : result_shape_(result_shape), result_nnz_(result_nnz),
        state_(std::move(state)) {}
#endif

  void update_impl_(index<> result_shape, offset_t result_nnz) {
    result_shape_ = result_shape;
    result_nnz_ = result_nnz;
  }

private:
  index<> result_shape_;
  offset_t result_nnz_;

#ifdef SPBLAS_ENABLE_ONEMKL_SYCL
public:
  __mkl::operation_state_t state_;
#endif

#ifdef SPBLAS_ENABLE_ARMPL
public:
  __armpl::operation_state_t state_;
#endif

#ifdef SPBLAS_ENABLE_AOCLSPARSE
public:
  __aoclsparse::operation_state_t state_;
#endif

#ifdef SPBLAS_ENABLE_CUSPARSE
public:
  __cusparse::operation_state_t state_;
#endif
};

} // namespace spblas
