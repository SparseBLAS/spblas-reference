#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <type_traits>

#include <hip/hip_runtime.h>
#include <rocsparse/rocsparse.h>

#include <spblas/detail/ranges.hpp>
#include <spblas/detail/triangular_types.hpp>
#include <spblas/detail/view_inspectors.hpp>

#include "exception.hpp"
#include "hip_allocator.hpp"
#include "types.hpp"

namespace spblas {
class triangular_solve_state_t {
public:
  triangular_solve_state_t()
      : triangular_solve_state_t(rocsparse::hip_allocator<char>{}) {}

  triangular_solve_state_t(rocsparse::hip_allocator<char> alloc)
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

  triangular_solve_state_t(rocsparse::hip_allocator<char> alloc,
                           rocsparse_handle handle)
      : alloc_(alloc), buffer_size_(0), workspace_(nullptr) {
    handle_ = handle_manager(handle, [](rocsparse_handle handle) {
      // it is provided by user, we do not delete it at all.
    });
  }

  ~triangular_solve_state_t() {
    alloc_.deallocate(workspace_);
  }

  template <matrix A, class Triangle, class DiagonalStorage, vector B, vector C>
    requires __detail::has_csr_base<A> &&
             __detail::has_contiguous_range_base<B> &&
             __ranges::contiguous_range<C>
  void triangular_solve(A&& a, Triangle uplo, DiagonalStorage diag, B&& b,
                        C&& c) {
    auto a_base = __detail::get_ultimate_base(a);
    auto b_base = __detail::get_ultimate_base(b);
    using matrix_type = decltype(a_base);
    using value_type = typename matrix_type::scalar_type;
    const auto diag_type = std::is_same_v<DiagonalStorage, explicit_diagonal_t>
                               ? rocsparse_diag_type_non_unit
                               : rocsparse_diag_type_unit;
    const auto fill_mode = std::is_same_v<Triangle, upper_triangle_t>
                               ? rocsparse_fill_mode_upper
                               : rocsparse_fill_mode_lower;

    auto a_descr = __rocsparse::create_rocsparse_handle(a_base);
    auto b_descr = __rocsparse::create_rocsparse_handle(b_base);
    auto c_descr = __rocsparse::create_rocsparse_handle(c);

    __rocsparse::throw_if_error(rocsparse_spmat_set_attribute(
        a_descr, rocsparse_spmat_fill_mode, &fill_mode, sizeof(fill_mode)));
    __rocsparse::throw_if_error(rocsparse_spmat_set_attribute(
        a_descr, rocsparse_spmat_diag_type, &diag_type, sizeof(diag_type)));
    value_type alpha = 1.0;
    size_t buffer_size = 0;
    auto handle = this->handle_.get();
    __rocsparse::throw_if_error(rocsparse_spsv(
        handle, rocsparse_operation_none, &alpha, a_descr, b_descr, c_descr,
        detail::rocsparse_data_type_v<value_type>, rocsparse_spsv_alg_default,
        rocsparse_spsv_stage_buffer_size, &buffer_size, nullptr));
    if (buffer_size > this->buffer_size_) {
      this->alloc_.deallocate(workspace_, this->buffer_size_);
      this->buffer_size_ = buffer_size;
      workspace_ = this->alloc_.allocate(buffer_size);
    }
    __rocsparse::throw_if_error(rocsparse_spsv(
        handle, rocsparse_operation_none, &alpha, a_descr, b_descr, c_descr,
        detail::rocsparse_data_type_v<value_type>, rocsparse_spsv_alg_default,
        rocsparse_spsv_stage_preprocess, &buffer_size, this->workspace_));
    __rocsparse::throw_if_error(rocsparse_spsv(
        handle, rocsparse_operation_none, &alpha, a_descr, b_descr, c_descr,
        detail::rocsparse_data_type_v<value_type>, rocsparse_spsv_alg_default,
        rocsparse_spsv_stage_compute, &buffer_size, this->workspace_));
    __rocsparse::throw_if_error(rocsparse_destroy_spmat_descr(a_descr));
    __rocsparse::throw_if_error(rocsparse_destroy_dnvec_descr(b_descr));
    __rocsparse::throw_if_error(rocsparse_destroy_dnvec_descr(c_descr));
  }

private:
  using handle_manager =
      std::unique_ptr<std::pointer_traits<rocsparse_handle>::element_type,
                      std::function<void(rocsparse_handle)>>;
  handle_manager handle_;
  rocsparse::hip_allocator<char> alloc_;
  std::uint64_t buffer_size_;
  char* workspace_;
};

template <matrix A, class Triangle, class DiagonalStorage, vector B, vector C>
  requires __detail::has_csr_base<A> &&
           __detail::has_contiguous_range_base<B> &&
           __ranges::contiguous_range<C>
void triangular_solve(triangular_solve_state_t& trisolve_handle, A&& a,
                      Triangle uplo, DiagonalStorage diag, B&& b, C&& c) {
  trisolve_handle.triangular_solve(a, uplo, diag, b, c);
}

} // namespace spblas
