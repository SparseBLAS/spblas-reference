#pragma once

#include <spblas/detail/index.hpp>
#include <spblas/detail/types.hpp>

#ifdef SPBLAS_ENABLE_ONEMKL
#include <spblas/vendor/mkl/operation_state_t.hpp>
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

  operation_info_t(index<> result_shape, index_t result_nnz)
      : result_shape_(result_shape), result_nnz_(result_nnz) {}

#ifdef SPBLAS_ENABLE_ONEMKL
  operation_info_t(index<> result_shape, index_t result_nnz,
                   __mkl::operation_state_t&& state)
      : result_shape_(result_shape), result_nnz_(result_nnz),
        state_(std::move(state)) {}
#endif

private:
  index<> result_shape_;
  index_t result_nnz_;

#ifdef SPBLAS_ENABLE_ONEMKL
public:
  __mkl::operation_state_t state_;
#endif
};

} // namespace spblas
