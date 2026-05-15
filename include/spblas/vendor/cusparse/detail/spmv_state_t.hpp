#pragma once

#include <cusparse.h>
#include <memory>

#include "abstract_operation_state.hpp"

namespace spblas {
namespace __cusparse {

class spmv_state_t : public abstract_operation_state_t {
public:
  spmv_state_t() = default;
  ~spmv_state_t() {
    if (a_descr_) {
      cusparseDestroySpMat(a_descr_);
    }
  }

  cusparseSpMatDescr_t a_descriptor() const {
    return a_descr_;
  }

  void set_a_descriptor(cusparseSpMatDescr_t descr) {
    a_descr_ = descr;
  }

private:
  cusparseSpMatDescr_t a_descr_ = nullptr;
};

} // namespace __cusparse
} // namespace spblas
