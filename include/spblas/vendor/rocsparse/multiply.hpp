#pragma once

#include <functional>
#include <memory>
#include <type_traits>

#include <hip/hip_runtime.h>
#include <rocsparse/rocsparse.h>

#include <spblas/detail/ranges.hpp>
#include <spblas/detail/view_inspectors.hpp>

#include "exception.hpp"
#include "hip_allocator.hpp"
#include "types.hpp"

namespace spblas {

class spmv_state_t {
public:
  spmv_state_t() : spmv_state_t(rocsparse::hip_allocator<char>{}) {}

  spmv_state_t(rocsparse::hip_allocator<char> alloc)
      : alloc_(alloc), buffer_size_(0), workspace_(nullptr) {
    rocsparse_handle handle;
    __rocsparse::throw_if_error(rocsparse_create_handle(&handle));
    if (auto stream = alloc.stream()) {
      rocsparse_set_stream(handle, stream);
    }
    handle_ = handle_manager(handle, [](rocsparse_handle handle) {
      __rocsparse::throw_if_error(rocsparse_destroy_handle(handle));
    });
  }

  spmv_state_t(rocsparse::hip_allocator<char> alloc, rocsparse_handle handle)
      : alloc_(alloc), buffer_size_(0), workspace_(nullptr) {
    handle_ = handle_manager(handle, [](rocsparse_handle handle) {
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

    rocsparse_spmat_descr mat;
    __rocsparse::throw_if_error(rocsparse_create_csr_descr(
        &mat, __backend::shape(a_base)[0], __backend::shape(a_base)[1],
        a_base.values().size(), a_base.rowptr().data(), a_base.colind().data(),
        a_base.values().data(),
        to_rocsparse_indextype<typename matrix_type::offset_type>(),
        to_rocsparse_indextype<typename matrix_type::index_type>(),
        rocsparse_index_base_zero, to_rocsparse_datatype<value_type>()));
    rocsparse_dnvec_descr vecb;
    rocsparse_dnvec_descr vecc;
    __rocsparse::throw_if_error(rocsparse_create_dnvec_descr(
        &vecb, b_base.size(), b_base.data(),
        to_rocsparse_datatype<typename input_type::value_type>()));
    __rocsparse::throw_if_error(rocsparse_create_dnvec_descr(
        &vecc, c.size(), c.data(),
        to_rocsparse_datatype<typename output_type::value_type>()));
    value_type alpha_val = alpha;
    value_type beta = 0.0;
    long unsigned int buffer_size = 0;
    // TODO: create a compute type for mixed precision computation
    __rocsparse::throw_if_error(rocsparse_spmv(
        handle, rocsparse_operation_none, &alpha_val, mat, vecb, &beta, vecc,
        to_rocsparse_datatype<value_type>(), rocsparse_spmv_alg_csr_stream,
        rocsparse_spmv_stage_buffer_size, &buffer_size, nullptr));
    // only allocate the new workspace when the requiring workspace larger than
    // current
    if (buffer_size > this->buffer_size_) {
      this->alloc_.deallocate(this->workspace_, buffer_size_);
      this->buffer_size_ = buffer_size;
      this->workspace_ = this->alloc_.allocate(buffer_size);
    }
    __rocsparse::throw_if_error(rocsparse_spmv(
        handle, rocsparse_operation_none, &alpha_val, mat, vecb, &beta, vecc,
        to_rocsparse_datatype<value_type>(), rocsparse_spmv_alg_csr_stream,
        rocsparse_spmv_stage_preprocess, &this->buffer_size_,
        this->workspace_));
    __rocsparse::throw_if_error(rocsparse_spmv(
        handle, rocsparse_operation_none, &alpha_val, mat, vecb, &beta, vecc,
        to_rocsparse_datatype<value_type>(), rocsparse_spmv_alg_csr_stream,
        rocsparse_spmv_stage_compute, &this->buffer_size_, this->workspace_));
    __rocsparse::throw_if_error(rocsparse_destroy_spmat_descr(mat));
    __rocsparse::throw_if_error(rocsparse_destroy_dnvec_descr(vecc));
    __rocsparse::throw_if_error(rocsparse_destroy_dnvec_descr(vecb));
  }

private:
  using handle_manager =
      std::unique_ptr<std::pointer_traits<rocsparse_handle>::element_type,
                      std::function<void(rocsparse_handle)>>;
  handle_manager handle_;
  rocsparse::hip_allocator<char> alloc_;
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
