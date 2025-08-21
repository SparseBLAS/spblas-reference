#pragma once

#include <type_traits>

#include <cuda_runtime.h>
#include <cusparse.h>

#include <spblas/detail/operation_info_t.hpp>
#include <spblas/detail/ranges.hpp>
#include <spblas/detail/view_inspectors.hpp>

#include "cuda_allocator.hpp"
#include "detail/cusparse_tensors.hpp"
#include "exception.hpp"
#include "types.hpp"

namespace spblas {
class sampled_multiply_state_t {
public:
  sampled_multiply_state_t()
      : sampled_multiply_state_t(cusparse::cuda_allocator<char>{}) {}

  sampled_multiply_state_t(cusparse::cuda_allocator<char> alloc)
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

  sampled_multiply_state_t(cusparse::cuda_allocator<char> alloc,
                           cusparseHandle_t handle)
      : alloc_(alloc), buffer_size_(0), workspace_(nullptr) {
    handle_ = handle_manager(handle, [](cusparseHandle_t handle) {
      // it is provided by user, we do not delete it at all.
    });
  }

  ~sampled_multiply_state_t() {
    alloc_.deallocate(workspace_);
  }

  template <matrix A, matrix B, matrix C>
    requires __detail::has_mdspan_matrix_base<A> &&
             __detail::has_mdspan_matrix_base<B> && __detail::has_csr_base<C>
  void sampled_multiply(A&& a, B&& b, C&& c) {
    auto a_base = __detail::get_ultimate_base(a);
    auto b_base = __detail::get_ultimate_base(b);
    auto c_base = __detail::get_ultimate_base(c);
    using matrix_type = decltype(a_base);
    using input_type = decltype(b_base);
    using output_type = std::remove_reference_t<decltype(c)>;
    using value_type = typename output_type::scalar_type;
    auto alpha_optional = __detail::get_scaling_factor(a, b);
    tensor_scalar_t<A> alpha = alpha_optional.value_or(1);
    value_type alpha_val = alpha;
    value_type beta = 0.0;

    auto a_descr = __cusparse::create_cusparse_handle(a_base);
    auto b_descr = __cusparse::create_cusparse_handle(b_base);
    auto c_descr = __cusparse::create_cusparse_handle(c_base);

    auto handle = this->handle_.get();
    long unsigned int buffer_size = 0;
    __cusparse::throw_if_error(cusparseSDDMM_bufferSize(
        handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha_val, a_descr, b_descr, &beta,
        c_descr, detail::cuda_data_type_v<value_type>,
        CUSPARSE_SDDMM_ALG_DEFAULT, &buffer_size));
    // only allocate the new workspace when the requiring workspace larger than
    // current
    if (buffer_size > buffer_size_) {
      this->buffer_size_ = buffer_size;
      alloc_.deallocate(this->workspace_);
      this->workspace_ = alloc_.allocate(buffer_size);
    }
    __cusparse::throw_if_error(cusparseSDDMM_preprocess(
        handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha_val, a_descr, b_descr, &beta,
        c_descr, detail::cuda_data_type_v<value_type>,
        CUSPARSE_SDDMM_ALG_DEFAULT, this->workspace_));

    __cusparse::throw_if_error(cusparseSDDMM(
        handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha_val, a_descr, b_descr, &beta,
        c_descr, detail::cuda_data_type_v<value_type>,
        CUSPARSE_SDDMM_ALG_DEFAULT, this->workspace_));
    __cusparse::throw_if_error(cusparseDestroyDnMat(a_descr));
    __cusparse::throw_if_error(cusparseDestroyDnMat(b_descr));
    __cusparse::throw_if_error(cusparseDestroySpMat(c_descr));
  }

private:
  using handle_manager =
      std::unique_ptr<std::pointer_traits<cusparseHandle_t>::element_type,
                      std::function<void(cusparseHandle_t)>>;
  handle_manager handle_;
  cusparse::cuda_allocator<char> alloc_;
  std::uint64_t buffer_size_;
  char* workspace_;
};

template <matrix A, matrix B, matrix C>
  requires __detail::has_mdspan_matrix_base<A> &&
           __detail::has_mdspan_matrix_base<B> && __detail::has_csr_base<C>
void sampled_multiply(sampled_multiply_state_t& sddmm_handle, A&& a, B&& b,
                      C&& c) {
  sddmm_handle.sampled_multiply(a, b, c);
}

} // namespace spblas
