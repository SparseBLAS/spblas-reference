#pragma once

#include <cusparse.h>
#include <memory>

#include "abstract_operation_state.hpp"

namespace spblas {
namespace __cusparse {

class spgemm_state_t : public abstract_operation_state_t {
public:
  spgemm_state_t() = default;
  ~spgemm_state_t() {
    if (a_descr_) {
      cusparseDestroySpMat(a_descr_);
    }
    if (b_descr_) {
      cusparseDestroySpMat(b_descr_);
    }
    if (c_descr_) {
      cusparseDestroySpMat(c_descr_);
    }
    if (spgemm_descr_) {
      cusparseSpGEMM_destroyDescr(spgemm_descr_);
    }
  }

  // Accessors for the descriptors
  cusparseSpMatDescr_t a_descriptor() const {
    return a_descr_;
  }
  cusparseDnVecDescr_t b_descriptor() const {
    return b_descr_;
  }
  cusparseDnVecDescr_t c_descriptor() const {
    return c_descr_;
  }
  cusparseSpGEMMDescr_t spgemm_descriptor() const {
    return spgemm_descr_;
  }

  // Setters for the descriptors
  void set_a_descriptor(cusparseSpMatDescr_t descr) {
    a_descr_ = descr;
  }
  void set_b_descriptor(cusparseDnVecDescr_t descr) {
    b_descr_ = descr;
  }
  void set_c_descriptor(cusparseDnVecDescr_t descr) {
    c_descr_ = descr;
  }
  void set_spgemm_descriptor(cusparseSpGEMMDescr_t descr) {
    spgemm_descr_ = descr;
  }

private:
  cusparseSpMatDescr_t a_descr_ = nullptr;
  cusparseSpMatDescr_t b_descr_ = nullptr;
  cusparseSpMatDescr_t c_descr_ = nullptr;
  cusparseSpGEMMDescr_t spgemm_descr_ = nullptr;
};

} // namespace __cusparse
} // namespace spblas
