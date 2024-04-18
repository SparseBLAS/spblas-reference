#pragma once

#include <spblas/detail/index.hpp>
#include <spblas/detail/types.hpp>

#ifdef SPBLAS_ENABLE_ONEMKL
#include <spblas/vendor/mkl/operation_state_t.hpp>
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

  operation_info_t(index<> result_shape, index_t result_nnz)
      : result_shape_(result_shape), result_nnz_(result_nnz) {}

#ifdef SPBLAS_ENABLE_ONEMKL
  operation_info_t(index<> result_shape, index_t result_nnz,
                   __mkl::operation_state_t&& state)
      : result_shape_(result_shape), result_nnz_(result_nnz),
        state_(std::move(state)) {}
#endif
#ifdef SPBLAS_ENABLE_CUSPARSE
  operation_info_t() : prepared_(false), step_(0), wait_allocation_(false) {}
#endif
private:
  index<> result_shape_;
  index_t result_nnz_;

#ifdef SPBLAS_ENABLE_ONEMKL
public:
  __mkl::operation_state_t state_;
#endif
#ifdef SPBLAS_ENABLE_CUSPARSE
private:
  std::vector<size_t> workspace_size_;
  std::vector<void*> workspace_;
  bool prepared_;
  bool wait_allocation_;
  int step_;

public:
  /**
   * get the workspace_size vector
   */
  const std::vector<size_t>& get_workspace_requirement() const {
    return workspace_size_;
  }

  /**
   * get the workspace pointer vector
   */
  const std::vector<void*>& get_workspace() const {
    return workspace_;
  }

  /**
   * get the last workspace size
   */
  size_t get_last_workspace_requirement() const {
    return workspace_size_.back();
  }

  /**
   * get the current step
   */
  int get_step() const {
    return step_;
  }

  /**
   * set the workspace size for this current step and increase the step by 1.
   */
  void set_last_workspace_requirement(size_t size) {
    workspace_size_.push_back(size);
    wait_allocation_ = true;
    step_++;
  }

  /**
   * set the matrix storage information
   */
  void set_matrix(index<> result_shape, index_t result_nnz) {
    result_shape_ = result_shape;
    result_nnz_ = result_nnz;
  }

  /**
   * set the workspace pointer
   */
  void set_last_workspace(void* workspace) {
    workspace_.push_back(workspace);
    wait_allocation_ = false;
  }

  /**
   * indicate the preparation is finished.
   */
  void finish_preparation() {
    prepared_ = true;
  }

  /**
   * check whether the workspace preparation is finished or not.
   */
  bool is_prepared() const {
    return prepared_;
  }

  __cusparse::operation_state_t state;
#endif
};

} // namespace spblas
