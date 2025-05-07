#pragma once

#include <functional>
#include <memory>
#include <type_traits>

#include <hip/hip_runtime.h>
#include <rocsparse/rocsparse.h>

#include <spblas/detail/ranges.hpp>
#include <spblas/detail/view_inspectors.hpp>

#include "descriptor.hpp"
#include "exception.hpp"
#include "hip_allocator.hpp"
#include "types.hpp"

namespace spblas {
namespace __rocsparse {

template <typename T>
T create_null_matrix() {
  return {nullptr, nullptr, nullptr, index<index_t>{0, 0}, 0};
}

} // namespace __rocsparse

class spgemm_state_t {
public:
  spgemm_state_t() : spgemm_state_t(rocsparse::hip_allocator<char>{}) {}

  spgemm_state_t(rocsparse::hip_allocator<char> alloc)
      : alloc_(alloc), buffer_size_(0), workspace_(nullptr), result_nnz_(0),
        result_shape_(0, 0) {
    rocsparse_handle handle;
    __rocsparse::throw_if_error(rocsparse_create_handle(&handle));
    if (auto stream = alloc.stream()) {
      rocsparse_set_stream(handle, stream);
    }
    handle_ = handle_manager(handle, [](rocsparse_handle handle) {
      __rocsparse::throw_if_error(rocsparse_destroy_handle(handle));
    });
  }

  spgemm_state_t(rocsparse::hip_allocator<char> alloc, rocsparse_handle handle)
      : alloc_(alloc), buffer_size_(0), workspace_(nullptr), result_nnz_(0),
        result_shape_(0, 0) {
    handle_ = handle_manager(handle, [](rocsparse_handle handle) {
      // it is provided by user, we do not delete it at all.
    });
  }

  ~spgemm_state_t() {
    alloc_.deallocate(this->workspace_, this->buffer_size_);
    __rocsparse::throw_if_error(rocsparse_destroy_spmat_descr(this->mat_a_));
    __rocsparse::throw_if_error(rocsparse_destroy_spmat_descr(this->mat_b_));
    __rocsparse::throw_if_error(rocsparse_destroy_spmat_descr(this->mat_c_));
    __rocsparse::throw_if_error(rocsparse_destroy_spmat_descr(this->mat_d_));
  }

  auto result_shape() {
    return this->result_shape_;
  }

  auto result_nnz() {
    return this->result_nnz_;
  }

  template <matrix A, matrix B, matrix C, matrix D>
    requires __detail::has_csr_base<A> && __detail::has_csr_base<B> &&
             __detail::is_csr_view_v<C> && __detail::has_csr_base<D>
  void multiply_compute(A&& a, B&& b, C&& c, D&& d) {
    auto a_base = __detail::get_ultimate_base(a);
    auto b_base = __detail::get_ultimate_base(b);
    auto d_base = __detail::get_ultimate_base(d);
    using matrix_type = decltype(a_base);
    using input_type = decltype(b_base);
    using output_type = std::remove_reference_t<decltype(c)>;
    using value_type = typename matrix_type::scalar_type;

    size_t buffer_size = 0;

    auto alpha_optional = __detail::get_scaling_factor(a, b);
    value_type alpha = alpha_optional.value_or(1);
    auto beta_optional = __detail::get_scaling_factor(d);
    value_type beta = beta_optional.value_or(1);
    auto handle = this->handle_.get();
    // Create sparse matrix A in CSR format
    this->mat_a_ = __rocsparse::create_matrix_descr(a_base);
    this->mat_b_ = __rocsparse::create_matrix_descr(b_base);
    this->mat_c_ = __rocsparse::create_matrix_descr(c);
    this->mat_d_ = __rocsparse::create_matrix_descr(d_base);
    //--------------------------------------------------------------------------
    // SpGEMM Computation
    // ask buffer_size bytes for external memory
    __rocsparse::throw_if_error(rocsparse_spgemm(
        handle, rocsparse_operation_none, rocsparse_operation_none, &alpha,
        this->mat_a_, this->mat_b_, &beta, this->mat_d_, this->mat_c_,
        to_rocsparse_datatype<value_type>(), rocsparse_spgemm_alg_default,
        rocsparse_spgemm_stage_buffer_size, &buffer_size, nullptr));
    if (buffer_size > this->buffer_size_) {
      this->alloc_.deallocate(workspace_, this->buffer_size_);
      this->buffer_size_ = buffer_size;
      workspace_ = this->alloc_.allocate(buffer_size);
    }
    __rocsparse::throw_if_error(rocsparse_spgemm(
        handle, rocsparse_operation_none, rocsparse_operation_none, &alpha,
        this->mat_a_, this->mat_b_, &beta, this->mat_d_, this->mat_c_,
        to_rocsparse_datatype<value_type>(), rocsparse_spgemm_alg_default,
        rocsparse_spgemm_stage_nnz, &this->buffer_size_, this->workspace_));
    // get matrix C non-zero entries and size
    int64_t c_num_rows;
    int64_t c_num_cols;
    __rocsparse::throw_if_error(rocsparse_spmat_get_size(
        this->mat_c_, &c_num_rows, &c_num_cols, &this->result_nnz_));
    this->result_shape_ = index<index_t>(c_num_rows, c_num_cols);
  }

  template <matrix A, matrix B, matrix C, matrix D>
    requires __detail::has_csr_base<A> && __detail::has_csr_base<B> &&
             __detail::is_csr_view_v<C> && __detail::has_csr_base<D>
  void multiply_fill(A&& a, B&& b, C&& c, D&& d) {
    auto a_base = __detail::get_ultimate_base(a);
    auto b_base = __detail::get_ultimate_base(b);
    using matrix_type = decltype(a_base);
    using input_type = decltype(b_base);
    using output_type = std::remove_reference_t<decltype(c)>;
    using value_type = typename matrix_type::scalar_type;

    auto alpha_optional = __detail::get_scaling_factor(a, b);
    tensor_scalar_t<A> alpha = alpha_optional.value_or(1);
    value_type alpha_val = alpha;
    auto beta_optional = __detail::get_scaling_factor(d);
    value_type beta = beta_optional.value_or(1);

    __rocsparse::throw_if_error(rocsparse_csr_set_pointers(
        this->mat_c_, c.rowptr().data(), c.colind().data(), c.values().data()));

    __rocsparse::throw_if_error(rocsparse_spgemm(
        handle_.get(), rocsparse_operation_none, rocsparse_operation_none,
        &alpha, this->mat_a_, this->mat_b_, &beta, this->mat_d_, this->mat_c_,
        to_rocsparse_datatype<value_type>(), rocsparse_spgemm_alg_default,
        rocsparse_spgemm_stage_compute, &this->buffer_size_, workspace_));
  }

  template <matrix A, matrix B, matrix C, matrix D>
    requires __detail::has_csr_base<A> && __detail::has_csr_base<B> &&
             __detail::is_csr_view_v<C> && __detail::has_csr_base<D>
  void multiply_symbolic_fill(A&& a, B&& b, C&& c, D&& d) {
    auto a_base = __detail::get_ultimate_base(a);
    auto b_base = __detail::get_ultimate_base(b);
    auto d_base = __detail::get_ultimate_base(d);
    using matrix_type = decltype(a_base);
    using input_type = decltype(b_base);
    using output_type = std::remove_reference_t<decltype(c)>;
    using value_type = typename matrix_type::scalar_type;

    auto alpha_optional = __detail::get_scaling_factor(a, b);
    value_type alpha = alpha_optional.value_or(1);
    auto beta_optional = __detail::get_scaling_factor(d);
    value_type beta = beta_optional.value_or(1);

    __rocsparse::throw_if_error(rocsparse_csr_set_pointers(
        this->mat_c_, c.rowptr().data(), c.colind().data(), c.values().data()));

    __rocsparse::throw_if_error(rocsparse_spgemm(
        this->handle_.get(), rocsparse_operation_none, rocsparse_operation_none,
        &alpha, this->mat_a_, this->mat_b_, &beta, this->mat_d_, this->mat_c_,
        to_rocsparse_datatype<value_type>(), rocsparse_spgemm_alg_default,
        rocsparse_spgemm_stage_symbolic, &this->buffer_size_,
        this->workspace_));
  }

  template <matrix A, matrix B, matrix C, matrix D>
    requires __detail::has_csr_base<A> && __detail::has_csr_base<B> &&
             __detail::is_csr_view_v<C> && __detail::has_csr_base<D>
  void multiply_numeric(A&& a, B&& b, C&& c, D&& d) {
    auto a_base = __detail::get_ultimate_base(a);
    auto b_base = __detail::get_ultimate_base(b);
    auto d_base = __detail::get_ultimate_base(d);
    using matrix_type = decltype(a_base);
    using input_type = decltype(b_base);
    using output_type = std::remove_reference_t<decltype(c)>;
    using value_type = typename matrix_type::scalar_type;

    auto alpha_optional = __detail::get_scaling_factor(a, b);
    tensor_scalar_t<A> alpha = alpha_optional.value_or(1);
    value_type alpha_val = alpha;
    auto beta_optional = __detail::get_scaling_factor(d);
    value_type beta = beta_optional.value_or(1);

    // Update the pointer from the matrix but they must contains the same
    // sparsity as the previous call.
    __rocsparse::throw_if_error(rocsparse_csr_set_pointers(
        this->mat_a_, a_base.rowptr().data(), a_base.colind().data(),
        a_base.values().data()));
    __rocsparse::throw_if_error(rocsparse_csr_set_pointers(
        this->mat_b_, b_base.rowptr().data(), b_base.colind().data(),
        b_base.values().data()));
    __rocsparse::throw_if_error(rocsparse_csr_set_pointers(
        this->mat_c_, c.rowptr().data(), c.colind().data(), c.values().data()));
    if (d_base.values().data()) {
      // when it is still a null matrix, we can not use set pointer function
      __rocsparse::throw_if_error(rocsparse_csr_set_pointers(
          this->mat_d_, d_base.rowptr().data(), d_base.colind().data(),
          d_base.values().data()));
    }
    __rocsparse::throw_if_error(rocsparse_spgemm(
        this->handle_.get(), rocsparse_operation_none, rocsparse_operation_none,
        &alpha, this->mat_a_, this->mat_b_, &beta, this->mat_d_, this->mat_c_,
        to_rocsparse_datatype<value_type>(), rocsparse_spgemm_alg_default,
        rocsparse_spgemm_stage_numeric, &this->buffer_size_, this->workspace_));
  }

private:
  using handle_manager =
      std::unique_ptr<std::pointer_traits<rocsparse_handle>::element_type,
                      std::function<void(rocsparse_handle)>>;
  handle_manager handle_;
  rocsparse::hip_allocator<char> alloc_;
  long unsigned int buffer_size_;
  char* workspace_;
  index<index_t> result_shape_;
  index_t result_nnz_;
  rocsparse_spmat_descr mat_a_;
  rocsparse_spmat_descr mat_b_;
  rocsparse_spmat_descr mat_c_;
  rocsparse_spmat_descr mat_d_;
};

template <matrix A, matrix B, matrix C>
  requires __detail::has_csr_base<A> && __detail::has_csr_base<B> &&
           __detail::is_csr_view_v<C>
void multiply_inspect(spgemm_state_t& spgemm_handle, A&& a, B&& b, C&& c) {}

template <matrix A, matrix B, matrix C, matrix D>
  requires __detail::has_csr_base<A> && __detail::has_csr_base<B> &&
           __detail::is_csr_view_v<C> && __detail::has_csr_base<D>
void multiply_compute(spgemm_state_t& spgemm_handle, A&& a, B&& b, C&& c,
                      D&& d) {
  spgemm_handle.multiply_compute(a, b, c, d);
}

template <matrix A, matrix B, matrix C, matrix D>
  requires __detail::has_csr_base<A> && __detail::has_csr_base<B> &&
           __detail::is_csr_view_v<C> && __detail::has_csr_base<D>
void multiply_fill(spgemm_state_t& spgemm_handle, A&& a, B&& b, C&& c, D&& d) {
  spgemm_handle.multiply_fill(a, b, c, d);
}

template <matrix A, matrix B, matrix C, matrix D>
  requires __detail::has_csr_base<A> && __detail::has_csr_base<B> &&
           __detail::is_csr_view_v<C> && __detail::has_csr_base<D>
void multiply_symbolic_compute(spgemm_state_t& spgemm_handle, A&& a, B&& b,
                               C&& c, D&& d) {
  spgemm_handle.multiply_compute(a, b, c, d);
}

template <matrix A, matrix B, matrix C, matrix D>
  requires __detail::has_csr_base<A> && __detail::has_csr_base<B> &&
           __detail::is_csr_view_v<C> && __detail::has_csr_base<D>
void multiply_symbolic_fill(spgemm_state_t& spgemm_handle, A&& a, B&& b, C&& c,
                            D&& d) {
  spgemm_handle.multiply_symbolic_fill(a, b, c, d);
}

template <matrix A, matrix B, matrix C, matrix D>
  requires __detail::has_csr_base<A> && __detail::has_csr_base<B> &&
           __detail::is_csr_view_v<C> && __detail::has_csr_base<D>
void multiply_numeric(spgemm_state_t& spgemm_handle, A&& a, B&& b, C&& c,
                      D&& d) {
  spgemm_handle.multiply_numeric(a, b, c, d);
}

// the followings support C = A*B by giving null D matrix.
template <matrix A, matrix B, matrix C>
  requires __detail::has_csr_base<A> && __detail::has_csr_base<B> &&
           __detail::is_csr_view_v<C>
void multiply_compute(spgemm_state_t& spgemm_handle, A&& a, B&& b, C&& c) {
  auto d = __rocsparse::create_null_matrix<std::remove_reference_t<C>>();
  spgemm_handle.multiply_compute(a, b, c, scaled(0.0, d));
}

template <matrix A, matrix B, matrix C>
  requires __detail::has_csr_base<A> && __detail::has_csr_base<B> &&
           __detail::is_csr_view_v<C>
void multiply_fill(spgemm_state_t& spgemm_handle, A&& a, B&& b, C&& c) {
  auto d = __rocsparse::create_null_matrix<std::remove_reference_t<C>>();
  spgemm_handle.multiply_fill(a, b, c, scaled(0.0, d));
}

template <matrix A, matrix B, matrix C>
  requires __detail::has_csr_base<A> && __detail::has_csr_base<B> &&
           __detail::is_csr_view_v<C>
void multiply_symbolic_compute(spgemm_state_t& spgemm_handle, A&& a, B&& b,
                               C&& c) {
  auto d = __rocsparse::create_null_matrix<std::remove_reference_t<C>>();
  spgemm_handle.multiply_compute(a, b, c, scaled(0.0, d));
}

template <matrix A, matrix B, matrix C>
  requires __detail::has_csr_base<A> && __detail::has_csr_base<B> &&
           __detail::is_csr_view_v<C>
void multiply_symbolic_fill(spgemm_state_t& spgemm_handle, A&& a, B&& b,
                            C&& c) {
  auto d = __rocsparse::create_null_matrix<std::remove_reference_t<C>>();
  spgemm_handle.multiply_symbolic_fill(a, b, c, scaled(0.0, d));
}

template <matrix A, matrix B, matrix C>
  requires __detail::has_csr_base<A> && __detail::has_csr_base<B> &&
           __detail::is_csr_view_v<C>
void multiply_numeric(spgemm_state_t& spgemm_handle, A&& a, B&& b, C&& c) {
  auto d = __rocsparse::create_null_matrix<std::remove_reference_t<C>>();
  spgemm_handle.multiply_numeric(a, b, c, scaled(0.0, d));
}

} // namespace spblas
