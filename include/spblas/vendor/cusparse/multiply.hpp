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

class spmv_state_t {
public:
  spmv_state_t() : spmv_state_t(cusparse::cuda_allocator<char>{}) {}

  spmv_state_t(cusparse::cuda_allocator<char> alloc)
      : alloc_(alloc), buffer_size_(0), workspace_(nullptr) {
    cusparseHandle_t handle;
    __cusparse::throw_if_error(cusparseCreate(&handle));
    if (auto stream = alloc.stream()) {
      cusparseSetStream(handle, stream);
    }
    handle_ = handle_manager(handle, [](cusparseHandle_t handle) {
      __cusparse::throw_if_error(cusparseDestroy(handle));
    });
  }

  spmv_state_t(cusparse::cuda_allocator<char> alloc, cusparseHandle_t handle)
      : alloc_(alloc), buffer_size_(0), workspace_(nullptr) {
    handle_ = handle_manager(handle, [](cusparseHandle_t handle) {
      // it is provided by user, we do not delete it at all.
    });
  }

  ~spmv_state_t() {
    alloc_.deallocate(workspace_, buffer_size_);
  }

  template <matrix A, vector B, vector C>
    requires __detail::has_csr_base<A> &&
             __detail::has_contiguous_range_base<B> &&
             __ranges::contiguous_range<C>
  void multiply(A&& a, B&& b, C&& c) {
    auto a_base = __detail::get_ultimate_base(a);
    auto b_base = __detail::get_ultimate_base(b);
    using matrix_type = decltype(a_base);
    using input_type = decltype(b_base);
    using output_type = std::remove_reference_t<decltype(c)>;
    using value_type = typename matrix_type::scalar_type;

    auto alpha_optional = __detail::get_scaling_factor(a, b);
    tensor_scalar_t<A> alpha = alpha_optional.value_or(1);
    auto handle = this->handle_.get();

    auto mat = __cusparse::create_matrix_descr(a_base);
    auto vecb = __cusparse::create_vector_descr(b_base);
    auto vecc = __cusparse::create_vector_descr(c);

    value_type alpha_val = alpha;
    value_type beta = 0.0;
    long unsigned int buffer_size = 0;
    // TODO: create a compute type for mixed precision computation
    __cusparse::throw_if_error(cusparseSpMV_bufferSize(
        handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha_val, mat, vecb, &beta,
        vecc, to_cuda_datatype<value_type>(), CUSPARSE_SPMV_ALG_DEFAULT,
        &buffer_size));
    // only allocate the new workspace when the requiring workspace larger than
    // current
    if (buffer_size > this->buffer_size_) {
      this->alloc_.deallocate(this->workspace_, buffer_size_);
      this->buffer_size_ = buffer_size;
      this->workspace_ = this->alloc_.allocate(buffer_size);
    }

    __cusparse::throw_if_error(
        cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha_val, mat,
                     vecb, &beta, vecc, to_cuda_datatype<value_type>(),
                     CUSPARSE_SPMV_ALG_DEFAULT, this->workspace_));
    __cusparse::throw_if_error(cusparseDestroySpMat(mat));
    __cusparse::throw_if_error(cusparseDestroyDnVec(vecc));
    __cusparse::throw_if_error(cusparseDestroyDnVec(vecb));
  }

private:
  using handle_manager =
      std::unique_ptr<std::pointer_traits<cusparseHandle_t>::element_type,
                      std::function<void(cusparseHandle_t)>>;
  handle_manager handle_;
  cusparse::cuda_allocator<char> alloc_;
  long unsigned int buffer_size_;
  char* workspace_;
};

template <matrix A, vector B, vector C>
  requires __detail::has_csr_base<A> &&
           __detail::has_contiguous_range_base<B> &&
           __ranges::contiguous_range<C>
void multiply(spmv_state_t& spmv_state, A&& a, B&& b, C&& c) {
  spmv_state.multiply(a, b, c);
}

} // namespace spblas
