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
// Defines the following APIs for SpGEMM:
//
//  C = op(A) * op(B)
//
//  where A,B and C are sparse matrices of CSR format
//

namespace spblas {

template <matrix A, matrix B, matrix C>
  requires(__detail::has_csr_base<A>) &&
          (__detail::has_csr_base<B>) && __detail::is_csr_view_v<C>
operation_info_t multiply_compute(A&& a, B&& b, C&& c) {
  log_trace("");
  auto a_base = __detail::get_ultimate_base(a);
  auto b_base = __detail::get_ultimate_base(b);

  using T = tensor_scalar_t<C>;
  using I = tensor_index_t<C>;
  using O = tensor_offset_t<C>;

  aoclsparse_matrix csrA = nullptr;
  aoclsparse_mat_descr descrA;
  aoclsparse_status status = aoclsparse_create_mat_descr(&descrA);
  if (status != aoclsparse_status_success) {
    fmt::print("\t descr creation failed\n");
  }
  aoclsparse_set_mat_type(descrA, aoclsparse_matrix_type_general);
  aoclsparse_index_base indexingA = aoclsparse_index_base_zero;
  aoclsparse_operation opA = aoclsparse_operation_none;

  const index_t a_nrows = __backend::shape(a_base)[0];
  const index_t a_ncols = __backend::shape(a_base)[1];

  aoclsparse_matrix csrB = nullptr;
  aoclsparse_mat_descr descrB;
  status = aoclsparse_create_mat_descr(&descrB);
  if (status != aoclsparse_status_success) {
    fmt::print("\t descr creation failed\n");
  }

  aoclsparse_set_mat_type(descrB, aoclsparse_matrix_type_general);
  aoclsparse_index_base indexingB = aoclsparse_index_base_zero;
  aoclsparse_operation opB = aoclsparse_operation_none;

  const index_t b_nrows = __backend::shape(b_base)[0];
  const index_t b_ncols = __backend::shape(b_base)[1];

  aoclsparse_matrix csrC = nullptr;
  aoclsparse_index_base indexingC = aoclsparse_index_base_zero;
  index_t c_nrows = 0, c_ncols = 0;
  offset_t c_nnz = -1;
  offset_t* c_rowptr = nullptr;
  index_t* c_colind = nullptr;
  T* c_values = nullptr;

  const aoclsparse_int nnzA = a_base.rowptr().data()[a_nrows];
  const aoclsparse_int nnzB = b_base.rowptr().data()[b_nrows];

  status = __aoclsparse::aoclsparse_create_csr(
      &csrA, indexingA, a_nrows, a_ncols, nnzA, a_base.rowptr().data(),
      a_base.colind().data(), a_base.values().data());
  if (status != aoclsparse_status_success) {
    fmt::print("\t csr matrix A creation failed\n");
  }
  __aoclsparse::aoclsparse_create_csr(
      &csrB, indexingB, b_nrows, b_ncols, nnzB, b_base.rowptr().data(),
      b_base.colind().data(), b_base.values().data());
  if (status != aoclsparse_status_success) {
    fmt::print("\t csr matrix B creation failed\n");
  }

  aoclsparse_request request = aoclsparse_stage_nnz_count;
  status =
      aoclsparse_sp2m(opA, descrA, csrA, opB, descrB, csrB, request, &csrC);
  if (status != aoclsparse_status_success) {
    fmt::print("\t SpGEMM nnz_count failed\n");
  }

  __aoclsparse::aoclsparse_export_csr(csrC, &indexingC, &c_nrows, &c_ncols,
                                      &c_nnz, &c_rowptr, &c_colind, &c_values);
  offset_t c_ind = indexingC == aoclsparse_index_base_zero ? 0 : 1;
  c_nnz = c_nnz - c_ind;
  log_info("computed c_nnz = %d", c_nnz);

  // Check: csrA and csrB destroyed when the operation_info destructor is
  // called?

  return operation_info_t{
      index<>{__backend::shape(c)[0], __backend::shape(c)[1]}, c_nnz,
      __aoclsparse::operation_state_t{csrA, csrB, csrC}};
}

template <matrix A, matrix B, matrix C>
  requires(__detail::has_csr_base<A>) &&
          (__detail::has_csr_base<B>) && __detail::is_csr_view_v<C>
void multiply_fill(operation_info_t& info, A&& a, B&& b, C&& c) {
  log_trace("");

  auto a_base = __detail::get_ultimate_base(a);
  auto b_base = __detail::get_ultimate_base(b);
  auto c_base = __detail::get_ultimate_base(c);

  using T = tensor_scalar_t<C>;
  using I = tensor_index_t<C>;
  using O = tensor_offset_t<C>;

  auto alpha_optional = __detail::get_scaling_factor(a, b);
  tensor_scalar_t<A> alpha = alpha_optional.value_or(1);

  aoclsparse_matrix csrA = info.state_.a_handle;
  aoclsparse_matrix csrB = info.state_.b_handle;
  aoclsparse_matrix csrC = info.state_.c_handle;
  offset_t c_nnz = info.result_nnz();

  aoclsparse_mat_descr descrA;
  aoclsparse_status status = aoclsparse_create_mat_descr(&descrA);
  if (status != aoclsparse_status_success) {
    fmt::print("\t descr creation failed\n");
  }

  aoclsparse_set_mat_type(descrA, aoclsparse_matrix_type_general);
  aoclsparse_index_base indexingA = aoclsparse_index_base_zero;
  aoclsparse_operation opA = aoclsparse_operation_none;

  const index_t a_nrows = __backend::shape(a_base)[0];
  const index_t a_ncols = __backend::shape(a_base)[1];

  aoclsparse_mat_descr descrB;
  status = aoclsparse_create_mat_descr(&descrB);
  if (status != aoclsparse_status_success) {
    fmt::print("\t descr creation failed\n");
  }

  aoclsparse_set_mat_type(descrB, aoclsparse_matrix_type_general);
  aoclsparse_index_base indexingB = aoclsparse_index_base_zero;
  aoclsparse_operation opB = aoclsparse_operation_none;

  const index_t b_nrows = __backend::shape(b_base)[0];
  const index_t b_ncols = __backend::shape(b_base)[1];

  aoclsparse_index_base indexingC = aoclsparse_index_base_zero;
  index_t c_nrows = __backend::shape(c_base)[0];
  index_t c_ncols = __backend::shape(c_base)[1];
  offset_t* c_rowptr = nullptr;
  index_t* c_colind = nullptr;
  T* c_values = nullptr;

  aoclsparse_request request = aoclsparse_stage_finalize;
  status =
      aoclsparse_sp2m(opA, descrA, csrA, opB, descrB, csrB, request, &csrC);
  if (status != aoclsparse_status_success) {
    fmt::print("\t SpGEMM failed: {}\n", (int) status);
  }

  status = __aoclsparse::aoclsparse_export_csr(csrC, &indexingC, &c_nrows,
                                               &c_ncols, &c_nnz, &c_rowptr,
                                               &c_colind, &c_values);
  if (status != aoclsparse_status_success) {
    fmt::print("\t exporting output matrix failed: {}\n", (int) status);
  }

  using T = tensor_scalar_t<C>;
  using I = tensor_index_t<C>;
  using O = tensor_offset_t<C>;

  // copy out of csrC arrays into C arrays
  std::copy(c_rowptr, c_rowptr + c_nrows + 1, c.rowptr().begin());
  std::copy(c_colind, c_colind + c_nnz, c.colind().begin());
  std::copy(c_values, c_values + c_nnz, c.values().begin());

  if (alpha_optional.has_value()) {
    scale(alpha, c);
  }
}

} // namespace spblas
