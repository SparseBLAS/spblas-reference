#pragma once

#include <type_traits>

#include <hip/hip_runtime.h>
#include <rocsparse/rocsparse.h>

#include <spblas/allocator.hpp>
#include <spblas/detail/ranges.hpp>
#include <spblas/detail/view_inspectors.hpp>

#include "types.hpp"

namespace spblas {

class spmv_state_t {
public:
  spmv_state_t(std::shared_ptr<const allocator> alloc)
      : alloc_(alloc), buffer_size_(0) {
    rocsparse_create_handle(&handle_);
  }

  ~spmv_state_t() {
    alloc_->free(workspace_);
    rocsparse_destroy_handle(handle_);
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

    rocsparse_spmat_descr mat;
    rocsparse_create_csr_descr(
        &mat, __backend::shape(a_base)[0], __backend::shape(a_base)[1],
        a_base.values().size(), a_base.rowptr().data(), a_base.colind().data(),
        a_base.values().data(),
        rocsparse_index_type<typename matrix_type::offset_type>(),
        rocsparse_index_type<typename matrix_type::index_type>(),
        rocsparse_index_base_zero, rocsparse_data_type<value_type>());

    rocsparse_dnvec_descr vecb, vecc;
    rocsparse_create_dnvec_descr(
        &vecb, b_base.size(), b_base.data(),
        rocsparse_data_type<typename input_type::value_type>());
    rocsparse_create_dnvec_descr(
        &vecc, c.size(), c.data(),
        rocsparse_data_type<typename output_type::value_type>());

    value_type alpha_val = alpha;
    value_type beta = 0.0;
    long unsigned int buffer_size = 0;
    // TODO: create a compute type for mixed precision computation
    rocsparse_spmv(handle_, rocsparse_operation_none, &alpha_val, mat, vecb,
                   &beta, vecc, rocsparse_data_type<value_type>(),
                   rocsparse_spmv_alg_csr_adaptive,
                   rocsparse_spmv_stage_buffer_size, &buffer_size, nullptr);
    // only allocate the new workspace when the requiring workspace larger than
    // current
    if (buffer_size > buffer_size_) {
      buffer_size_ = buffer_size;
      alloc_->free(workspace_);
      alloc_->alloc(&workspace_, buffer_size);
    }

    rocsparse_spmv(handle_, rocsparse_operation_none, &alpha_val, mat, vecb,
                   &beta, vecc, rocsparse_data_type<value_type>(),
                   rocsparse_spmv_alg_csr_adaptive,
                   rocsparse_spmv_stage_preprocess, &buffer_size, workspace_);
    rocsparse_spmv(handle_, rocsparse_operation_none, &alpha_val, mat, vecb,
                   &beta, vecc, rocsparse_data_type<value_type>(),
                   rocsparse_spmv_alg_csr_adaptive,
                   rocsparse_spmv_stage_compute, &buffer_size, workspace_);
    rocsparse_destroy_spmat_descr(mat);
    rocsparse_destroy_dnvec_descr(vecc);
    rocsparse_destroy_dnvec_descr(vecb);
  }

private:
  rocsparse_handle handle_;
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
