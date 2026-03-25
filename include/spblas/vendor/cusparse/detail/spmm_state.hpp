#pragma once

#include <cusparse.h>
#include <memory>

#include "abstract_operation_state.hpp"

namespace spblas {
namespace __cusparse {

class spmm_state_t : public abstract_operation_state_t {
public:
  spmm_state_t() = default;
  ~spmm_state_t() {
    if (a_descr_) {
      cusparseDestroySpMat(a_descr_);
    }
    if (x_descr_) {
      cusparseDestroyDnMat(x_descr_);
    }
    if (y_descr_) {
      cusparseDestroyDnMat(y_descr_);
    }
  }

  // Accessors for the descriptors
  cusparseSpMatDescr_t a_descriptor() const {
    return a_descr_;
  }
  cusparseDnMatescr_t x_descriptor() const {
    return x_descr_;
  }
  cusparseDnMatDescr_t y_descriptor() const {
    return y_descr_;
  }

  // Setters for the descriptors
  void set_a_descriptor(cusparseSpMatDescr_t descr) {
    a_descr_ = descr;
  }
  void set_x_descriptor(cusparseDnMatDescr_t descr) {
    x_descr_ = descr;
  }
  void set_y_descriptor(cusparseDnMatDescr_t descr) {
    y_descr_ = descr;
  }

private:
  cusparseSpMatDescr_t a_descr_ = nullptr;
  cusparseDnMatDescr_t x_descr_ = nullptr;
  cusparseDnMatDescr_t y_descr_ = nullptr;
};

} // namespace __cusparse
} // namespace spblas
