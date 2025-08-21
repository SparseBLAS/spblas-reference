#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <type_traits>

#include <cuda_runtime.h>
#include <cusparse.h>

#include <spblas/detail/ranges.hpp>
#include <spblas/detail/triangular_types.hpp>
#include <spblas/detail/view_inspectors.hpp>

#include "cuda_allocator.hpp"
#include "detail/cusparse_tensors.hpp"
#include "exception.hpp"
#include "types.hpp"

namespace spblas {
class triangular_solve_state_t {
public:
  triangular_solve_state_t()
      : triangular_solve_state_t(cusparse::cuda_allocator<char>{}) {}

  triangular_solve_state_t(cusparse::cuda_allocator<char> alloc)
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

  triangular_solve_state_t(cusparse::cuda_allocator<char> alloc,
                           cusparseHandle_t handle)
      : alloc_(alloc), buffer_size_(0), workspace_(nullptr) {
    handle_ = handle_manager(handle, [](cusparseHandle_t handle) {
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
    // the following needs to be non-const because cusparseSpMatSetAttribute
    // only accept void*
    auto diag_type = std::is_same_v<DiagonalStorage, explicit_diagonal_t>
                         ? CUSPARSE_DIAG_TYPE_NON_UNIT
                         : CUSPARSE_DIAG_TYPE_UNIT;
    auto fill_mode = std::is_same_v<Triangle, upper_triangle_t>
                         ? CUSPARSE_FILL_MODE_UPPER
                         : CUSPARSE_FILL_MODE_LOWER;

    auto a_descr = __cusparse::create_cusparse_handle(a_base);
    auto b_descr = __cusparse::create_cusparse_handle(b_base);
    auto c_descr = __cusparse::create_cusparse_handle(c);

    __cusparse::throw_if_error(cusparseSpMatSetAttribute(
        a_descr, CUSPARSE_SPMAT_FILL_MODE, &fill_mode, sizeof(fill_mode)));
    __cusparse::throw_if_error(cusparseSpMatSetAttribute(
        a_descr, CUSPARSE_SPMAT_DIAG_TYPE, &diag_type, sizeof(diag_type)));
    value_type alpha = 1.0;
    size_t buffer_size = 0;
    auto handle = this->handle_.get();
    cusparseSpSVDescr_t descr;
    cusparseSpSV_createDescr(&descr);
    __cusparse::throw_if_error(cusparseSpSV_bufferSize(
        handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, a_descr, b_descr,
        c_descr, detail::cuda_data_type_v<value_type>,
        CUSPARSE_SPSV_ALG_DEFAULT, descr, &buffer_size));
    if (buffer_size > this->buffer_size_) {
      this->alloc_.deallocate(workspace_, this->buffer_size_);
      this->buffer_size_ = buffer_size;
      workspace_ = this->alloc_.allocate(buffer_size);
    }
    __cusparse::throw_if_error(cusparseSpSV_analysis(
        handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, a_descr, b_descr,
        c_descr, detail::cuda_data_type_v<value_type>,
        CUSPARSE_SPSV_ALG_DEFAULT, descr, this->workspace_));
    __cusparse::throw_if_error(cusparseSpSV_solve(
        handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, a_descr, b_descr,
        c_descr, detail::cuda_data_type_v<value_type>,
        CUSPARSE_SPSV_ALG_DEFAULT, descr));
    __cusparse::throw_if_error(cusparseDestroySpMat(a_descr));
    __cusparse::throw_if_error(cusparseDestroyDnVec(b_descr));
    __cusparse::throw_if_error(cusparseDestroyDnVec(c_descr));
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

template <matrix A, class Triangle, class DiagonalStorage, vector B, vector C>
  requires __detail::has_csr_base<A> &&
           __detail::has_contiguous_range_base<B> &&
           __ranges::contiguous_range<C>
void triangular_solve(triangular_solve_state_t& trisolve_handle, A&& a,
                      Triangle uplo, DiagonalStorage diag, B&& b, C&& c) {
  trisolve_handle.triangular_solve(a, uplo, diag, b, c);
}

} // namespace spblas
