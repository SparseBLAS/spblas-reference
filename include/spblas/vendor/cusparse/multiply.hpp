#pragma once

#include <type_traits>

#include <cuda_runtime.h>
#include <cusparse.h>

#include <spblas/allocator.hpp>
#include <spblas/detail/ranges.hpp>
#include <spblas/detail/view_inspectors.hpp>

#include "types.hpp"

namespace spblas {

class spmv_state_t {
public:
  spmv_state_t(std::shared_ptr<const allocator> alloc)
      : alloc_(alloc), buffer_size_(0) {
    cusparseCreate(&handle_);
  }

  ~spmv_state_t() {
    alloc_->free(workspace_);
    cusparseDestroy(handle_);
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

    cusparseSpMatDescr_t mat;
    cusparseCreateCsr(&mat, __backend::shape(a_base)[0],
                      __backend::shape(a_base)[1], a_base.values().size(),
                      a_base.rowptr().data(), a_base.colind().data(),
                      a_base.values().data(),
                      cusparse_index_type<typename matrix_type::offset_type>(),
                      cusparse_index_type<typename matrix_type::index_type>(),
                      CUSPARSE_INDEX_BASE_ZERO, cuda_data_type<value_type>());

    cusparseDnVecDescr_t vecb, vecc;
    cusparseCreateDnVec(&vecb, b_base.size(), b_base.data(),
                        cuda_data_type<typename input_type::value_type>());
    cusparseCreateDnVec(&vecc, c.size(), c.data(),
                        cuda_data_type<typename output_type::value_type>());

    value_type alpha_val = alpha;
    value_type beta = 0.0;
    long unsigned int buffer_size = 0;
    // TODO: create a compute type for mixed precision computation
    cusparseSpMV_bufferSize(handle_, CUSPARSE_OPERATION_NON_TRANSPOSE,
                            &alpha_val, mat, vecb, &beta, vecc,
                            cuda_data_type<value_type>(),
                            CUSPARSE_SPMV_ALG_DEFAULT, &buffer_size);
    // only allocate the new workspace when the requiring workspace larger than
    // current
    if (buffer_size > buffer_size_) {
      buffer_size_ = buffer_size;
      alloc_->free(workspace_);
      alloc_->alloc(&workspace_, buffer_size);
    }

    cusparseSpMV(handle_, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha_val, mat,
                 vecb, &beta, vecc, cuda_data_type<value_type>(),
                 CUSPARSE_SPMV_ALG_DEFAULT, workspace_);
    cusparseDestroyDnVec(vecc);
    cusparseDestroyDnVec(vecb);
  }

private:
  cusparseHandle_t handle_;
  std::shared_ptr<const allocator> alloc_;
  long unsigned int buffer_size_;
  void* workspace_;
};

template <matrix A, vector B, vector C>
  requires __detail::has_csr_base<A> &&
           __detail::has_contiguous_range_base<B> &&
           __ranges::contiguous_range<C>
void multiply(spmv_state_t& spmv_handle, A&& a, B&& b, C&& c) {
  spmv_handle.multiply(a, b, c);
}

} // namespace spblas
