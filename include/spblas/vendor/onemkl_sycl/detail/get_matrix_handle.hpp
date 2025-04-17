#pragma once

#include <oneapi/mkl.hpp>

#include <spblas/detail/log.hpp>
#include <spblas/detail/operation_info_t.hpp>
#include <spblas/detail/ranges.hpp>
#include <spblas/detail/view_inspectors.hpp>
#include <spblas/views/matrix_opt.hpp>

#include <spblas/vendor/onemkl_sycl/detail/create_matrix_handle.hpp>

namespace spblas {

namespace __mkl {

template <matrix M>
oneapi::mkl::sparse::matrix_handle_t
get_matrix_handle(sycl::queue& q, M&& m,
                  oneapi::mkl::sparse::matrix_handle_t handle = nullptr) {
  if constexpr (__detail::is_matrix_opt_v<decltype(m)>) {
    log_trace("using A as matrix_opt");

    if (m.matrix_handle_ == nullptr) {
      m.matrix_handle_ = create_matrix_handle(q, m.base());
    }

    return m.matrix_handle_;
  } else if constexpr (__detail::has_base<M>) {
    return get_matrix_handle(q, m.base(), handle);
  } else if (handle != nullptr) {
    log_trace("using A from operation_info_t");

    return handle;
  } else {
    log_trace("using A as csr_base");

    return create_matrix_handle(q, m);
  }
}

} // namespace __mkl

} // namespace spblas
