#pragma once

#include <spblas/detail/index.hpp>
#include <spblas/detail/types.hpp>

namespace spblas {

struct vendor_t;

class operation_info_t {
public:
  auto result_shape() {
    return result_shape_;
  }

  auto result_nnz() {
    return result_nnz_;
  }

  auto get_vendor_info() {
    return vendor_info_;
  }

  void add_vendor_info(std::shared_ptr<vendor_t>& vendor_info) {
    vendor_info_ = vendor_info;
  }

  operation_info_t(index<> result_shape, index_t result_nnz)
      : result_shape_(result_shape), result_nnz_(result_nnz),
        vendor_info_(nullptr) {}

private:
  index<> result_shape_;
  index_t result_nnz_;
  std::shared_ptr<vendor_t> vendor_info_;
};

} // namespace spblas
