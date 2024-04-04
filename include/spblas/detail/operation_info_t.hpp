#pragma once

#include <spblas/detail/index.hpp>
#include <spblas/detail/types.hpp>

#include <cusparse.h>

namespace spblas {

class operation_info_t {
public:
  auto result_shape() {
    return result_shape_;
  }

  auto result_nnz() {
    return result_nnz_;
  }

  cusparseSpGEMMDescr_t spgemm_descr;

  operation_info_t(index<> result_shape, index_t result_nnz)
      : result_shape_(result_shape), result_nnz_(result_nnz) {}

private:
  index<> result_shape_;
  index_t result_nnz_;
};

} // namespace spblas
