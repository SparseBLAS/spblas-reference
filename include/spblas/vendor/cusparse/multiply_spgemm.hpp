#pragma once

#include <functional>
#include <memory>
#include <type_traits>

#include <cuda_runtime.h>
#include <cusparse.h>

#include <spblas/detail/ranges.hpp>
#include <spblas/detail/view_inspectors.hpp>

#include "cuda_allocator.hpp"
#include "exception.hpp"
#include "types.hpp"

namespace spblas {

class spgemm_state_t {
public:
  spgemm_state_t() : spgemm_state_t(cusparse::cuda_allocator<char>{}) {}

  spgemm_state_t(cusparse::cuda_allocator<char> alloc)
      : alloc_(alloc), buffer_size_1_(0), buffer_size_2_(0),
        workspace_1_(nullptr), workspace_2_(nullptr), result_nnz_(0),
        result_shape_(0, 0) {
    cusparseHandle_t handle;
    __cusparse::throw_if_error(cusparseCreate(&handle));
    if (auto stream = alloc.stream()) {
      cusparseSetStream(handle, stream);
    }
    handle_ = handle_manager(handle, [](cusparseHandle_t handle) {
      __cusparse::throw_if_error(cusparseDestroy(handle));
    });
    __cusparse::throw_if_error(cusparseSpGEMM_createDescr(&descr_));
  }

  spgemm_state_t(cusparse::cuda_allocator<char> alloc, cusparseHandle_t handle)
      : alloc_(alloc), buffer_size_1_(0), buffer_size_2_(0),
        workspace_1_(nullptr), workspace_2_(nullptr), result_nnz_(0),
        result_shape_(0, 0) {
    handle_ = handle_manager(handle, [](cusparseHandle_t handle) {
      // it is provided by user, we do not delete it at all.
    });
    __cusparse::throw_if_error(cusparseSpGEMM_createDescr(&descr_));
  }

  ~spgemm_state_t() {
    alloc_.deallocate(workspace_1_, buffer_size_1_);
    alloc_.deallocate(workspace_2_, buffer_size_2_);
    __cusparse::throw_if_error(cusparseDestroySpMat(mat_a_));
    __cusparse::throw_if_error(cusparseDestroySpMat(mat_b_));
    __cusparse::throw_if_error(cusparseDestroySpMat(mat_c_));
    __cusparse::throw_if_error(cusparseSpGEMM_destroyDescr(descr_));
  }

  auto result_shape() {
    return this->result_shape_;
  }

  auto result_nnz() {
    return this->result_nnz_;
  }

  template <matrix A, matrix B, matrix C>
    requires __detail::has_csr_base<A> && __detail::has_csr_base<B> &&
             __detail::is_csr_view_v<C>
  void multiply_compute(A&& a, B&& b, C&& c) {
    auto a_base = __detail::get_ultimate_base(a);
    auto b_base = __detail::get_ultimate_base(b);
    using matrix_type = decltype(a_base);
    using input_type = decltype(b_base);
    using output_type = std::remove_reference_t<decltype(c)>;
    using value_type = typename matrix_type::scalar_type;
    size_t buffer_size = 0;

    auto alpha_optional = __detail::get_scaling_factor(a, b);
    value_type alpha = alpha_optional.value_or(1);
    value_type beta = 1;
    auto handle = this->handle_.get();
    // Create sparse matrix A in CSR format
    __cusparse::throw_if_error(cusparseCreateCsr(
        &mat_a_, __backend::shape(a_base)[0], __backend::shape(a_base)[1],
        a_base.values().size(), a_base.rowptr().data(), a_base.colind().data(),
        a_base.values().data(),
        to_cusparse_indextype<typename matrix_type::offset_type>(),
        to_cusparse_indextype<typename matrix_type::index_type>(),
        CUSPARSE_INDEX_BASE_ZERO, to_cuda_datatype<value_type>()));
    __cusparse::throw_if_error(cusparseCreateCsr(
        &mat_b_, __backend::shape(b_base)[0], __backend::shape(b_base)[1],
        b_base.values().size(), b_base.rowptr().data(), b_base.colind().data(),
        b_base.values().data(),
        to_cusparse_indextype<typename matrix_type::offset_type>(),
        to_cusparse_indextype<typename matrix_type::index_type>(),
        CUSPARSE_INDEX_BASE_ZERO, to_cuda_datatype<value_type>()));
    __cusparse::throw_if_error(cusparseCreateCsr(
        &mat_c_, __backend::shape(a)[0], __backend::shape(b)[1], 0,
        c.rowptr().data(), NULL, NULL,
        to_cusparse_indextype<typename matrix_type::offset_type>(),
        to_cusparse_indextype<typename matrix_type::index_type>(),
        CUSPARSE_INDEX_BASE_ZERO, to_cuda_datatype<value_type>()));

    // ask bufferSize1 bytes for external memory
    size_t buffer_size_1 = 0;
    __cusparse::throw_if_error(cusparseSpGEMM_workEstimation(
        handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, mat_a_, mat_b_, &beta, mat_c_,
        to_cuda_datatype<value_type>(), CUSPARSE_SPGEMM_DEFAULT, this->descr_,
        &buffer_size_1, NULL));
    if (buffer_size_1 > this->buffer_size_1_) {
      this->alloc_.deallocate(this->workspace_1_, buffer_size_1_);
      this->buffer_size_1_ = buffer_size_1;
      this->workspace_1_ = this->alloc_.allocate(buffer_size_1);
    }
    // inspect the matrices A and B to understand the memory requirement for
    // the next step
    __cusparse::throw_if_error(cusparseSpGEMM_workEstimation(
        handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, mat_a_, mat_b_, &beta, mat_c_,
        to_cuda_datatype<value_type>(), CUSPARSE_SPGEMM_DEFAULT, this->descr_,
        &buffer_size_1, this->workspace_1_));

    // ask buffer_size_2 bytes for external memory
    size_t buffer_size_2 = 0;
    __cusparse::throw_if_error(cusparseSpGEMM_compute(
        handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, mat_a_, mat_b_, &beta, mat_c_,
        to_cuda_datatype<value_type>(), CUSPARSE_SPGEMM_DEFAULT, this->descr_,
        &buffer_size_2, NULL));
    if (buffer_size_2 > this->buffer_size_2_) {
      this->alloc_.deallocate(this->workspace_2_, buffer_size_2_);
      this->buffer_size_2_ = buffer_size_2;
      this->workspace_2_ = this->alloc_.allocate(buffer_size_2);
    }

    // compute the intermediate product of A * B
    cusparseSpGEMM_compute(
        handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, mat_a_, mat_b_, &beta, mat_c_,
        to_cuda_datatype<value_type>(), CUSPARSE_SPGEMM_DEFAULT, this->descr_,
        &buffer_size_2, this->workspace_2_);
    // get matrix C non-zero entries c_nnz
    int64_t c_num_rows, c_num_cols, c_nnz;
    cusparseSpMatGetSize(mat_c_, &c_num_rows, &c_num_cols, &c_nnz);
    this->result_nnz_ = c_nnz;
    this->result_shape_ = index<index_t>(c_num_rows, c_num_cols);
  }

  // C = AB
  template <matrix A, matrix B, matrix C>
    requires __detail::has_csr_base<A> && __detail::has_csr_base<B> &&
             __detail::is_csr_view_v<C>
  void multiply_fill(A&& a, B&& b, C&& c) {
    auto a_base = __detail::get_ultimate_base(a);
    auto b_base = __detail::get_ultimate_base(b);
    using matrix_type = decltype(a_base);
    using input_type = decltype(b_base);
    using output_type = std::remove_reference_t<decltype(c)>;
    using value_type = typename matrix_type::scalar_type;

    auto alpha_optional = __detail::get_scaling_factor(a, b);
    value_type alpha = alpha_optional.value_or(1);
    value_type beta = 1;
    auto handle = this->handle_.get();
    __cusparse::throw_if_error(cusparseCsrSetPointers(
        mat_c_, c.rowptr().data(), c.colind().data(), c.values().data()));

    __cusparse::throw_if_error(cusparseSpGEMM_copy(
        handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, mat_a_, mat_b_, &beta, mat_c_,
        to_cuda_datatype<value_type>(), CUSPARSE_SPGEMM_DEFAULT, this->descr_));
  }

private:
  using handle_manager =
      std::unique_ptr<std::pointer_traits<cusparseHandle_t>::element_type,
                      std::function<void(cusparseHandle_t)>>;
  handle_manager handle_;
  cusparse::cuda_allocator<char> alloc_;
  size_t buffer_size_1_;
  size_t buffer_size_2_;
  char* workspace_1_;
  char* workspace_2_;
  index<index_t> result_shape_;
  index_t result_nnz_;
  cusparseSpMatDescr_t mat_a_;
  cusparseSpMatDescr_t mat_b_;
  cusparseSpMatDescr_t mat_c_;
  cusparseSpGEMMDescr_t descr_;
};

template <matrix A, matrix B, matrix C>
  requires __detail::has_csr_base<A> && __detail::has_csr_base<B> &&
           __detail::is_csr_view_v<C>
void multiply_inspect(spgemm_state_t& spgemm_handle, A&& a, B&& b, C&& c) {}

template <matrix A, matrix B, matrix C>
  requires __detail::has_csr_base<A> && __detail::has_csr_base<B> &&
           __detail::is_csr_view_v<C>
void multiply_compute(spgemm_state_t& spgemm_handle, A&& a, B&& b, C&& c) {
  spgemm_handle.multiply_compute(a, b, c);
}

template <matrix A, matrix B, matrix C>
  requires __detail::has_csr_base<A> && __detail::has_csr_base<B> &&
           __detail::is_csr_view_v<C>
void multiply_fill(spgemm_state_t& spgemm_handle, A&& a, B&& b, C&& c) {
  spgemm_handle.multiply_fill(a, b, c);
}

} // namespace spblas
