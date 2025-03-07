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
// Defines the following APIs for SpMM:
//
//  Y = alpha * op(A) * X
//
//  where A is a sparse matrices of CSR format and
//  X/Y are dense matrices of row_major format
//

namespace spblas {

template <matrix A, matrix X, matrix Y>
  requires __detail::has_csr_base<A> && __detail::has_mdspan_matrix_base<X> &&
           __detail::is_matrix_instantiation_of_mdspan_v<Y> &&
           std::is_same_v<
               typename __detail::ultimate_base_type_t<X>::layout_type,
               __mdspan::layout_right> &&
           std::is_same_v<typename std::remove_cvref_t<Y>::layout_type,
                          __mdspan::layout_right>
void multiply(A&& a, X&& x, Y&& y) {
  log_trace("");
  auto a_base = __detail::get_ultimate_base(a);
  auto x_base = __detail::get_ultimate_base(x);
  auto y_base = __detail::get_ultimate_base(y);

  using T = tensor_scalar_t<A>;
  using I = tensor_index_t<A>;
  using O = tensor_offset_t<A>;

  auto alpha_optional = __detail::get_scaling_factor(a, x);
  tensor_scalar_t<A> alpha = alpha_optional.value_or(1);

  aoclsparse_matrix csrA = nullptr;
  aoclsparse_mat_descr descr;
  aoclsparse_status status = aoclsparse_create_mat_descr(&descr);
  if (status != aoclsparse_status_success) {
    fmt::print("\t Descr creation failed: {}\n", (int) status);
  }
  aoclsparse_set_mat_type(descr, aoclsparse_matrix_type_general);
  aoclsparse_index_base indexing = aoclsparse_index_base_zero;
  aoclsparse_operation opA = aoclsparse_operation_none;
  aoclsparse_order layout = aoclsparse_order_row;

  const index_t a_nrows = __backend::shape(a_base)[0];
  const index_t a_ncols = __backend::shape(a_base)[1];
  const index_t nrhs = x_base.extent(1);
  const index_t ldx = x_base.extent(1);
  const index_t ldy = y_base.extent(1);

  const aoclsparse_int nnz = a_base.rowptr().data()[a_nrows];

  status = spblas::__aoclsparse::aoclsparse_create_csr(
      &csrA, indexing, a_nrows, a_ncols, nnz, a_base.rowptr().data(),
      a_base.colind().data(), a_base.values().data());
  if (status != aoclsparse_status_success) {
    fmt::print("\t csr matrix creation failed: {}\n", (int) status);
  }

  T beta = static_cast<T>(0.0);
  status = spblas::__aoclsparse::aoclsparse_csrmm(
      opA, alpha, csrA, descr, layout, x_base.data_handle(), nrhs, ldx, beta,
      y.data_handle(), ldy);
  if (status != aoclsparse_status_success) {
    fmt::print("\t SpMM failed: {}\n", (int) status);
  }
  aoclsparse_destroy(&csrA);
}
} // namespace spblas
