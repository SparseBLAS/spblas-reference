#pragma once

#include <stdexcept>
#include <type_traits>

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <spblas/detail/operation_info_t.hpp>
#include <spblas/detail/ranges.hpp>
#include <spblas/detail/view_inspectors.hpp>

#include "cuda_allocator.hpp"
#include "detail/cusparse_tensors.hpp"
#include "exception.hpp"
#include "types.hpp"

namespace spblas {

class simple_operation_state_t {
public:
  simple_operation_state_t()
      : simple_operation_state_t(cusparse::cuda_allocator<char>{}) {}

  simple_operation_state_t(cusparse::cuda_allocator<char> alloc)
      : alloc_(alloc) {
    cublasHandle_t handle;
    __cusparse::throw_if_error(cublasCreate(&handle));
    if (auto stream = alloc.stream()) {
      __cusparse::throw_if_error(cublasSetStream(handle, stream));
    }
    handle_ = handle_manager(handle, [](cublasHandle_t handle) {
      __cusparse::throw_if_error(cublasDestroy(handle));
    });
  }

  simple_operation_state_t(cusparse::cuda_allocator<char> alloc,
                           cublasHandle_t handle)
      : alloc_(alloc) {
    handle_ = handle_manager(handle, [](cublasHandle_t handle) {
      // it is provided by user, we do not delete it at all.
    });
  }

  template <matrix A>
    requires __detail::has_csr_base<A>
  void scale(typename std::remove_reference_t<A>::scalar_type val, A&& a) {
    auto a_base = __detail::get_ultimate_base(a);
    using matrix_type = decltype(a_base);
    using value_type = typename matrix_type::scalar_type;
    if constexpr (std::is_same_v<value_type, float>) {
      __cusparse::throw_if_error(
          cublasSscal(handle_.get(), static_cast<int>(a_base.values().size()),
                      &val, a_base.values().data(), 1));
    } else if constexpr (std::is_same_v<value_type, double>) {
      __cusparse::throw_if_error(
          cublasDscal(handle_.get(), static_cast<int>(a_base.values().size()),
                      &val, a_base.values().data(), 1));
    } else {
      throw std::runtime_error("not implemented");
    }
  }

  template <matrix A>
    requires __detail::has_csr_base<A>
  typename std::remove_reference_t<A>::scalar_type matrix_inf_norm(A&& a) {
    auto a_base = __detail::get_ultimate_base(a);
    using matrix_type = decltype(a_base);
    using value_type = typename matrix_type::scalar_type;
    using index_type = typename matrix_type::index_type;
    value_type result = 0;
    // very slow implementation by calling cublas row by row
    for (int i = 0; i < __backend::shape(a_base)[0]; i++) {
      value_type tmp = 0;
      index_type start, end;
      __cusparse::throw_if_error(cudaMemcpy(&start, a_base.rowptr().data() + i,
                                            sizeof(index_type),
                                            cudaMemcpyDeviceToHost));
      __cusparse::throw_if_error(
          cudaMemcpy(&end, a_base.rowptr().data() + i + 1, sizeof(index_type),
                     cudaMemcpyDeviceToHost));
      if constexpr (std::is_same_v<value_type, float>) {
        __cusparse::throw_if_error(cublasSasum(handle_.get(), end - start,
                                               a_base.values().data() + start,
                                               1, &tmp));
      } else if constexpr (std::is_same_v<value_type, double>) {
        __cusparse::throw_if_error(cublasDasum(handle_.get(), end - start,
                                               a_base.values().data() + start,
                                               1, &tmp));
      } else {
        throw std::runtime_error("not implemented");
      }
      result = std::max(result, tmp);
    }
    return result;
  }

  template <matrix A>
    requires __detail::has_csr_base<A>
  typename std::remove_reference_t<A>::scalar_type matrix_frob_norm(A&& a) {
    auto a_base = __detail::get_ultimate_base(a);
    using matrix_type = decltype(a_base);
    using value_type = typename matrix_type::scalar_type;
    value_type result(0.0);
    if constexpr (std::is_same_v<value_type, float>) {
      __cusparse::throw_if_error(
          cublasSnrm2(handle_.get(), static_cast<int>(a_base.values().size()),
                      a_base.values().data(), 1, &result));
    } else if constexpr (std::is_same_v<value_type, double>) {
      __cusparse::throw_if_error(
          cublasDnrm2(handle_.get(), static_cast<int>(a_base.values().size()),
                      a_base.values().data(), 1, &result));
    } else {
      throw std::runtime_error("not implemented");
    }
    return result;
  }

private:
  using handle_manager =
      std::unique_ptr<std::pointer_traits<cublasHandle_t>::element_type,
                      std::function<void(cublasHandle_t)>>;
  handle_manager handle_;
  cusparse::cuda_allocator<char> alloc_;
};

using scale_state_t = simple_operation_state_t;
using matrix_inf_norm_state_t = simple_operation_state_t;
using matrix_frob_norm_state_t = simple_operation_state_t;

template <matrix A>
  requires __detail::has_csr_base<A>
void scale(scale_state_t& state,
           typename std::remove_reference_t<A>::scalar_type val, A&& a) {
  state.scale(val, a);
}

template <matrix A>
  requires __detail::has_csr_base<A>
typename std::remove_reference_t<A>::scalar_type
matrix_inf_norm(matrix_inf_norm_state_t& state, A&& a) {
  return state.matrix_inf_norm(a);
}

template <matrix A>
  requires __detail::has_csr_base<A>
typename std::remove_reference_t<A>::scalar_type
matrix_frob_norm(matrix_frob_norm_state_t& state, A&& a) {
  return state.matrix_frob_norm(a);
}

} // namespace spblas
