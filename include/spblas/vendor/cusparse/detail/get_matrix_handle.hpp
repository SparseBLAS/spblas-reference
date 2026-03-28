#pragma once

#include <cusparse.h>

#include <spblas/detail/log.hpp>
#include <spblas/detail/operation_info_t.hpp>
#include <spblas/detail/ranges.hpp>
#include <spblas/detail/view_inspectors.hpp>
#include <spblas/views/matrix_opt.hpp>

#include <spblas/vendor/cusparse/detail/create_matrix_handle.hpp>

namespace spblas {

namespace __cusparse {

template <matrix M>
cusparseSpMatDescr_t
get_matrix_handle(M&& m,
                  cusparseSpMatDescr_t handle = nullptr) {
  if constexpr (__detail::is_matrix_opt_v<decltype(m)>) {
    log_trace("using A as matrix_opt");

    if (m.matrix_handle_ == nullptr) {
      m.matrix_handle_ = create_matrix_handle(m.base());
    }

    return m.matrix_handle_;
  } else if constexpr (__detail::has_base<M>) {
    return get_matrix_handle(m.base(), handle);
  } else if (handle != nullptr) {
    log_trace("using A from operation_info_t");

    return handle;
  } else {
    log_trace("using A as csr_base");

    return create_matrix_handle(m);
  }
}

} // namespace __cusparse

} // namespace spblas
