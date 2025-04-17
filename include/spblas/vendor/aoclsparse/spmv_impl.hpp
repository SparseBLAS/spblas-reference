/*
 * Copyright (c) 2025      Advanced Micro Devices, Inc. All Rights reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */

#pragma once

#include "aoclsparse.h"
#include <cstdint>

#include "aocl_wrappers.hpp"
#include <fmt/core.h>
#include <spblas/detail/log.hpp>
#include <spblas/detail/operation_info_t.hpp>
#include <spblas/detail/ranges.hpp>
#include <spblas/detail/view_inspectors.hpp>

//
// Defines the following APIs for SpMV:
//
// y =alpha* op(A) * x
//
//  where A is a sparse matrices of CSR format and
//  x/y are dense vectors

namespace spblas {

template <matrix A, vector X, vector Y>
  requires(__detail::has_csr_base<A> || __detail::has_csc_base<A>) &&
          __detail::has_contiguous_range_base<X> &&
          __ranges::contiguous_range<Y>
void multiply(A&& a, X&& x, Y&& y) {
  log_trace("");
  auto a_base = __detail::get_ultimate_base(a);
  auto x_base = __detail::get_ultimate_base(x);

  aoclsparse_matrix csrA = __aoclsparse::create_matrix_handle(a_base);
  aoclsparse_operation opA = __aoclsparse::get_transpose(a);

  using T = tensor_scalar_t<A>;
  using I = tensor_index_t<A>;
  using O = tensor_offset_t<A>;

  auto alpha_optional = __detail::get_scaling_factor(a, x);
  T alpha = alpha_optional.value_or(1);

  aoclsparse_mat_descr descr = NULL;
  aoclsparse_status status = aoclsparse_create_mat_descr(&descr);
  if (status != aoclsparse_status_success) {
    fmt::print("\t descr creation failed\n");
  }
  aoclsparse_set_mat_type(descr, aoclsparse_matrix_type_general);
  aoclsparse_index_base indexing = aoclsparse_index_base_zero;

  // Do we need these two
  aoclsparse_set_mv_hint(csrA, opA, descr, 1);
  aoclsparse_optimize(csrA);

  T beta = static_cast<T>(0.0);
  status = __aoclsparse::aoclsparse_mv(opA, &alpha, csrA, descr,
                                       __ranges::data(x_base), &beta,
                                       __ranges::data(y));
  if (status != aoclsparse_status_success) {
    fmt::print("\t SpMV failed: {}\n", (int) status);
  }

  aoclsparse_destroy(&csrA);
  aoclsparse_destroy_mat_descr(descr);
}

} // namespace spblas
