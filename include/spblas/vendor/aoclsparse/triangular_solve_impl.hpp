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

#include <spblas/detail/triangular_types.hpp>

namespace spblas {

template <matrix A, class Triangle, class DiagonalStorage, vector B, vector X>
  requires __detail::has_csr_base<A> &&
           __detail::has_contiguous_range_base<B> &&
           __ranges::contiguous_range<X>
void triangular_solve(A&& a, Triangle uplo, DiagonalStorage diag, B&& b,
                      X&& x) {
  log_trace("");
  static_assert(std::is_same_v<Triangle, upper_triangle_t> ||
                std::is_same_v<Triangle, lower_triangle_t>);
  static_assert(std::is_same_v<DiagonalStorage, explicit_diagonal_t> ||
                std::is_same_v<DiagonalStorage, implicit_unit_diagonal_t>);

  auto a_base = __detail::get_ultimate_base(a);
  auto b_base = __detail::get_ultimate_base(b);

  using T = tensor_scalar_t<A>;
  using I = tensor_index_t<A>;
  using O = tensor_offset_t<A>;

  auto alpha_optional = __detail::get_scaling_factor(a, x);
  T alpha = alpha_optional.value_or(1);

  aoclsparse_matrix csrA = NULL;
  aoclsparse_mat_descr descr = NULL;
  aoclsparse_status status = aoclsparse_create_mat_descr(&descr);
  if (status != aoclsparse_status_success) {
    fmt::print("\t descr creation failed\n");
  }
  aoclsparse_set_mat_type(descr, aoclsparse_matrix_type_triangular);
  aoclsparse_index_base indexing = aoclsparse_index_base_zero;
  aoclsparse_operation opA = aoclsparse_operation_none;

  const index_t a_nrows = __backend::shape(a_base)[0];
  const index_t a_ncols = __backend::shape(a_base)[1];
  const aoclsparse_int nnz = a_base.rowptr().data()[a_nrows];

  status = __aoclsparse::aoclsparse_create_csr(
      &csrA, indexing, a_nrows, a_ncols, nnz, a_base.rowptr().data(),
      a_base.colind().data(), a_base.values().data());
  if (status != aoclsparse_status_success) {
    fmt::print("\t csr matrix creation failed: {}\n", (int) status);
  }
  if (std::is_same_v<Triangle, lower_triangle_t>) {
    aoclsparse_set_mat_fill_mode(descr, aoclsparse_fill_mode_lower);
  } else if (std::is_same_v<Triangle, upper_triangle_t>) {
    aoclsparse_set_mat_fill_mode(descr, aoclsparse_fill_mode_upper);
  }

  if (std::is_same_v<DiagonalStorage, explicit_diagonal_t>) {
    aoclsparse_set_mat_diag_type(descr, aoclsparse_diag_type_non_unit);
  } else if (std::is_same_v<DiagonalStorage, implicit_unit_diagonal_t>) {
    aoclsparse_set_mat_diag_type(descr, aoclsparse_diag_type_unit);
  }

  status = __aoclsparse::aoclsparse_trsv(
      opA, alpha, csrA, descr, __ranges::data(b_base), __ranges::data(x));
  if (status != aoclsparse_status_success) {
    fmt::print("\t triangular solve failed: {} \n", (int) status);
  }

  aoclsparse_destroy(&csrA);
  aoclsparse_destroy_mat_descr(descr);
}

} // namespace spblas
