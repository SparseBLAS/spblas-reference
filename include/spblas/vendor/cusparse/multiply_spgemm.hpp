#pragma once

#include <functional>
#include <memory>
#include <type_traits>

#include <cuda_runtime.h>
#include <cusparse.h>

#include <spblas/detail/ranges.hpp>
#include <spblas/detail/view_inspectors.hpp>

#include "cuda_allocator.hpp"
#include "descriptor.hpp"
#include "exception.hpp"
#include "types.hpp"

namespace spblas {

class spgemm_state_t {
public:
  spgemm_state_t() : spgemm_state_t(cusparse::cuda_allocator<char>{}) {}

  spgemm_state_t(cusparse::cuda_allocator<char> alloc)
      : alloc_(alloc), buffer_size_1_(0), buffer_size_2_(0), buffer_size_3_(0),
        buffer_size_4_(0), buffer_size_5_(0), workspace_1_(nullptr),
        workspace_2_(nullptr), workspace_3_(nullptr), workspace_4_(nullptr),
        workspace_5_(nullptr), result_nnz_(0), result_shape_(0, 0) {
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
    __cusparse::throw_if_error(cusparseDestroySpMat(mat_a_));
    __cusparse::throw_if_error(cusparseDestroySpMat(mat_b_));
    __cusparse::throw_if_error(cusparseDestroySpMat(mat_c_));
    mat_a_ = __cusparse::create_matrix_descr(a_base);
    mat_b_ = __cusparse::create_matrix_descr(b_base);
    mat_c_ = __cusparse::create_matrix_descr(c);

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

  template <matrix A, matrix B, matrix C>
    requires __detail::has_csr_base<A> && __detail::has_csr_base<B> &&
             __detail::is_csr_view_v<C>
  void multiply_symbolic_compute(A&& a, B&& b, C&& c) {
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
    __cusparse::throw_if_error(cusparseDestroySpMat(mat_a_));
    __cusparse::throw_if_error(cusparseDestroySpMat(mat_b_));
    __cusparse::throw_if_error(cusparseDestroySpMat(mat_c_));
    mat_a_ = __cusparse::create_matrix_descr(a_base);
    mat_b_ = __cusparse::create_matrix_descr(b_base);
    mat_c_ = __cusparse::create_matrix_descr(c);

    // ask bufferSize1 bytes for external memory
    size_t buffer_size_1 = 0;
    __cusparse::throw_if_error(cusparseSpGEMMreuse_workEstimation(
        handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE, mat_a_, mat_b_, mat_c_,
        CUSPARSE_SPGEMM_DEFAULT, this->descr_, &buffer_size_1, NULL));
    if (buffer_size_1 > this->buffer_size_1_) {
      this->alloc_.deallocate(this->workspace_1_, buffer_size_1_);
      this->buffer_size_1_ = buffer_size_1;
      this->workspace_1_ = this->alloc_.allocate(buffer_size_1);
    }
    // inspect the matrices A and B to understand the memory requirement for
    // the next step
    __cusparse::throw_if_error(cusparseSpGEMMreuse_workEstimation(
        handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE, mat_a_, mat_b_, mat_c_,
        CUSPARSE_SPGEMM_DEFAULT, this->descr_, &buffer_size_1,
        this->workspace_1_));

    // ask buffer_size_2/3/4 bytes for external memory
    size_t buffer_size_2 = 0;
    size_t buffer_size_3 = 0;
    size_t buffer_size_4 = 0;
    cusparseSpGEMMreuse_nnz(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                            CUSPARSE_OPERATION_NON_TRANSPOSE, mat_a_, mat_b_,
                            mat_c_, CUSPARSE_SPGEMM_DEFAULT, this->descr_,
                            &buffer_size_2, NULL, &buffer_size_3, NULL,
                            &buffer_size_4, NULL);
    if (buffer_size_2 > this->buffer_size_2_) {
      this->alloc_.deallocate(this->workspace_2_, buffer_size_2_);
      this->buffer_size_2_ = buffer_size_2;
      this->workspace_2_ = this->alloc_.allocate(buffer_size_2);
    }
    if (buffer_size_3 > this->buffer_size_3_) {
      this->alloc_.deallocate(this->workspace_3_, buffer_size_3_);
      this->buffer_size_3_ = buffer_size_3;
      this->workspace_3_ = this->alloc_.allocate(buffer_size_3);
    }
    if (buffer_size_4 > this->buffer_size_4_) {
      this->alloc_.deallocate(this->workspace_4_, buffer_size_4_);
      this->buffer_size_4_ = buffer_size_4;
      this->workspace_4_ = this->alloc_.allocate(buffer_size_4);
    }

    // compute nnz
    cusparseSpGEMMreuse_nnz(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                            CUSPARSE_OPERATION_NON_TRANSPOSE, mat_a_, mat_b_,
                            mat_c_, CUSPARSE_SPGEMM_DEFAULT, this->descr_,
                            &buffer_size_2, this->workspace_2_, &buffer_size_3,
                            this->workspace_3_, &buffer_size_4,
                            this->workspace_4_);
    // get matrix C non-zero entries c_nnz
    int64_t c_num_rows, c_num_cols, c_nnz;
    cusparseSpMatGetSize(mat_c_, &c_num_rows, &c_num_cols, &c_nnz);
    this->result_nnz_ = c_nnz;
    this->result_shape_ = index<index_t>(c_num_rows, c_num_cols);
  }

  template <matrix A, matrix B, matrix C>
    requires __detail::has_csr_base<A> && __detail::has_csr_base<B> &&
             __detail::is_csr_view_v<C>
  void multiply_symbolic_fill(A&& a, B&& b, C&& c) {
    auto a_base = __detail::get_ultimate_base(a);
    auto b_base = __detail::get_ultimate_base(b);
    using matrix_type = decltype(a_base);
    using input_type = decltype(b_base);
    using output_type = std::remove_reference_t<decltype(c)>;
    using value_type = typename matrix_type::scalar_type;

    auto alpha_optional = __detail::get_scaling_factor(a, b);
    value_type alpha = alpha_optional.value_or(1);
    value_type beta = 0;

    __cusparse::throw_if_error(cusparseCsrSetPointers(
        this->mat_c_, c.rowptr().data(), c.colind().data(), c.values().data()));

    auto handle = this->handle_.get();
    size_t buffer_size_5 = 0;
    cusparseSpGEMMreuse_copy(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                             CUSPARSE_OPERATION_NON_TRANSPOSE, mat_a_, mat_b_,
                             mat_c_, CUSPARSE_SPGEMM_DEFAULT, this->descr_,
                             &buffer_size_5, NULL);
    if (buffer_size_5 > this->buffer_size_5_) {
      this->alloc_.deallocate(this->workspace_5_, buffer_size_5_);
      this->buffer_size_5_ = buffer_size_5;
      this->workspace_5_ = this->alloc_.allocate(buffer_size_5);
    }
    cusparseSpGEMMreuse_copy(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                             CUSPARSE_OPERATION_NON_TRANSPOSE, mat_a_, mat_b_,
                             mat_c_, CUSPARSE_SPGEMM_DEFAULT, this->descr_,
                             &buffer_size_5, this->workspace_5_);
  }

  template <matrix A, matrix B, matrix C>
    requires __detail::has_csr_base<A> && __detail::has_csr_base<B> &&
             __detail::is_csr_view_v<C>
  void multiply_numeric(A&& a, B&& b, C&& c) {
    auto a_base = __detail::get_ultimate_base(a);
    auto b_base = __detail::get_ultimate_base(b);
    using matrix_type = decltype(a_base);
    using input_type = decltype(b_base);
    using output_type = std::remove_reference_t<decltype(c)>;
    using value_type = typename matrix_type::scalar_type;

    auto alpha_optional = __detail::get_scaling_factor(a, b);
    tensor_scalar_t<A> alpha = alpha_optional.value_or(1);
    value_type alpha_val = alpha;
    value_type beta = 0;

    auto handle = this->handle_.get();

    // Update the pointer from the matrix but they must contains the same
    // sparsity as the previous call.
    __cusparse::throw_if_error(
        cusparseCsrSetPointers(this->mat_a_, a_base.rowptr().data(),
                               a_base.colind().data(), a_base.values().data()));
    __cusparse::throw_if_error(
        cusparseCsrSetPointers(this->mat_b_, b_base.rowptr().data(),
                               b_base.colind().data(), b_base.values().data()));
    __cusparse::throw_if_error(cusparseCsrSetPointers(
        this->mat_c_, c.rowptr().data(), c.colind().data(), c.values().data()));
    cusparseSpGEMMreuse_compute(
        handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, mat_a_, mat_b_, &beta, mat_c_,
        to_cuda_datatype<value_type>(), CUSPARSE_SPGEMM_DEFAULT, this->descr_);
  }

private:
  using handle_manager =
      std::unique_ptr<std::pointer_traits<cusparseHandle_t>::element_type,
                      std::function<void(cusparseHandle_t)>>;
  handle_manager handle_;
  cusparse::cuda_allocator<char> alloc_;
  size_t buffer_size_1_;
  size_t buffer_size_2_;
  size_t buffer_size_3_;
  size_t buffer_size_4_;
  size_t buffer_size_5_;
  char* workspace_1_;
  char* workspace_2_;
  char* workspace_3_;
  char* workspace_4_;
  char* workspace_5_;
  index<index_t> result_shape_;
  index_t result_nnz_;
  cusparseSpMatDescr_t mat_a_ = nullptr;
  cusparseSpMatDescr_t mat_b_ = nullptr;
  cusparseSpMatDescr_t mat_c_ = nullptr;
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

template <matrix A, matrix B, matrix C>
  requires __detail::has_csr_base<A> && __detail::has_csr_base<B> &&
           __detail::is_csr_view_v<C>
void multiply_symbolic_compute(spgemm_state_t& spgemm_handle, A&& a, B&& b,
                               C&& c) {
  spgemm_handle.multiply_symbolic_compute(a, b, c);
}

template <matrix A, matrix B, matrix C>
  requires __detail::has_csr_base<A> && __detail::has_csr_base<B> &&
           __detail::is_csr_view_v<C>
void multiply_symbolic_fill(spgemm_state_t& spgemm_handle, A&& a, B&& b,
                            C&& c) {
  spgemm_handle.multiply_symbolic_fill(a, b, c);
}

template <matrix A, matrix B, matrix C>
  requires __detail::has_csr_base<A> && __detail::has_csr_base<B> &&
           __detail::is_csr_view_v<C>
void multiply_numeric(spgemm_state_t& spgemm_handle, A&& a, B&& b, C&& c) {
  spgemm_handle.multiply_numeric(a, b, c);
}

} // namespace spblas
