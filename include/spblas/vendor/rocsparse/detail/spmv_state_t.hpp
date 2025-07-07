#pragma once

#include <memory>
#include <rocsparse/rocsparse.h>

#include "../hip_allocator.hpp"
#include "abstract_operation_state.hpp"

namespace spblas {
namespace __rocsparse {

class spmv_state_t : public abstract_operation_state_t {
public:
  spmv_state_t() : spmv_state_t(rocsparse::hip_allocator<char>{}) {}

  spmv_state_t(rocsparse::hip_allocator<char> alloc)
      : alloc_(alloc), buffer_size_(0), workspace_(nullptr), a_descr_(nullptr),
        b_descr_(nullptr), c_descr_(nullptr) {}

  ~spmv_state_t() {
    if (workspace_) {
      alloc_.deallocate(workspace_, buffer_size_);
    }
    if (a_descr_) {
      rocsparse_destroy_spmat_descr(a_descr_);
    }
    if (b_descr_) {
      rocsparse_destroy_dnvec_descr(b_descr_);
    }
    if (c_descr_) {
      rocsparse_destroy_dnvec_descr(c_descr_);
    }
  }

  // Workspace management
  void* workspace() const {
    return workspace_;
  }
  size_t buffer_size() const {
    return buffer_size_;
  }

  void allocate_workspace(size_t size) {
    if (size > buffer_size_) {
      if (workspace_) {
        alloc_.deallocate(workspace_, buffer_size_);
      }
      buffer_size_ = size;
      workspace_ = alloc_.allocate(size);
    }
  }

  // Descriptor accessors
  rocsparse_spmat_descr a_descriptor() const {
    return a_descr_;
  }
  rocsparse_dnvec_descr b_descriptor() const {
    return b_descr_;
  }
  rocsparse_dnvec_descr c_descriptor() const {
    return c_descr_;
  }

  // Descriptor setters
  void set_a_descriptor(rocsparse_spmat_descr descr) {
    if (a_descr_) {
      rocsparse_destroy_spmat_descr(a_descr_);
    }
    a_descr_ = descr;
  }

  void set_b_descriptor(rocsparse_dnvec_descr descr) {
    if (b_descr_) {
      rocsparse_destroy_dnvec_descr(b_descr_);
    }
    b_descr_ = descr;
  }

  void set_c_descriptor(rocsparse_dnvec_descr descr) {
    if (c_descr_) {
      rocsparse_destroy_dnvec_descr(c_descr_);
    }
    c_descr_ = descr;
  }

private:
  rocsparse::hip_allocator<char> alloc_;
  size_t buffer_size_;
  char* workspace_;

  // Descriptors
  rocsparse_spmat_descr a_descr_;
  rocsparse_dnvec_descr b_descr_;
  rocsparse_dnvec_descr c_descr_;
};

} // namespace __rocsparse
} // namespace spblas
