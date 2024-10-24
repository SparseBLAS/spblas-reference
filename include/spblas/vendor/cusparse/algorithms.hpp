#pragma once

#include "add_impl.hpp"
#include "conversion_impl.hpp"
#include "filter_impl.hpp"
#include "multiply_impl.hpp"
#include "sddmm_impl.hpp"
#include "trisolve_impl.hpp"
#include <iostream>
#include <spblas/backend/concepts.hpp>

#include <cublas_v2.h>
#include <type_traits>

namespace spblas {

class simple_operation_handle_t {
public:
  simple_operation_handle_t(std::shared_ptr<const allocator> alloc)
      : alloc_(alloc) {
    cublasCreate(&handle_);
  }

  ~simple_operation_handle_t() {
    cublasDestroy(handle_);
  }

  template <matrix A>
    requires __detail::has_csr_base<A>
  void scale(typename std::remove_reference_t<A>::scalar_type val, A&& a) {
    auto a_base = __detail::get_ultimate_base(a);
    using matrix_type = decltype(a_base);
    using value_type = typename matrix_type::scalar_type;
    if constexpr (std::is_same_v<value_type, float>) {
      cublasSscal(handle_, static_cast<int>(a_base.values().size()), &val,
                  a_base.values().data(), 1);
    } else {
      cublasDscal(handle_, static_cast<int>(a_base.values().size()), &val,
                  a_base.values().data(), 1);
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
    for (int i = 0; i < __backend::shape(a_base)[0]; i++) {
      value_type tmp = 0;
      index_type start, end;
      cudaMemcpy(&start, a_base.rowptr().data() + i, sizeof(index_type),
                 cudaMemcpyDeviceToHost);
      cudaMemcpy(&end, a_base.rowptr().data() + i + 1, sizeof(index_type),
                 cudaMemcpyDeviceToHost);
      if constexpr (std::is_same_v<value_type, float>) {
        cublasSasum(handle_, end - start, a_base.values().data() + start, 1,
                    &tmp);
      } else {
        cublasDasum(handle_, end - start, a_base.values().data() + start, 1,
                    &tmp);
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
      cublasSnrm2(handle_, static_cast<int>(a_base.values().size()),
                  a_base.values().data(), 1, &result);
    } else {
      cublasDnrm2(handle_, static_cast<int>(a_base.values().size()),
                  a_base.values().data(), 1, &result);
    }
    return result;
  }

private:
  cublasHandle_t handle_;
  std::shared_ptr<const allocator> alloc_;
};

using scale_handle_t = simple_operation_handle_t;
using matrix_inf_norm_handle_t = simple_operation_handle_t;
using matrix_frob_norm_handle_t = simple_operation_handle_t;

template <matrix A>
  requires __detail::has_csr_base<A>
void scale(scale_handle_t& handle,
           typename std::remove_reference_t<A>::scalar_type val, A&& a) {
  handle.scale(val, a);
}

template <matrix A>
  requires __detail::has_csr_base<A>
typename std::remove_reference_t<A>::scalar_type
matrix_inf_norm(matrix_inf_norm_handle_t& handle, A&& a) {
  return handle.matrix_inf_norm(a);
}

template <matrix A>
  requires __detail::has_csr_base<A>
typename std::remove_reference_t<A>::scalar_type
matrix_frob_norm(matrix_frob_norm_handle_t& handle, A&& a) {
  return handle.matrix_frob_norm(a);
}

} // namespace spblas
