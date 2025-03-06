#pragma once

#include <stdlib.h> // aligned_alloc
#include <algorithm>
#include "mkl.h"

#include "mkl_wrappers.hpp"
#include <spblas/detail/operation_info_t.hpp>
#include <spblas/detail/ranges.hpp>
#include <spblas/detail/view_inspectors.hpp>
#include <spblas/detail/log.hpp>

//
// Defines the following APIs for SpGEMM:
//
//  C = op(A) * op(B)
//
//  where A,B and C are sparse matrices of CSR format
//
// operation_info_t multiply_inspect(A, B, C)
// void multiply_execute(operation_info_t, A, B, C)
//



namespace spblas {


template <matrix A, matrix B, matrix C>
  requires __detail::has_csr_base<A> && __detail::has_csr_base<B> &&
           __detail::is_csr_view_v<C>
operation_info_t multiply_compute(A&& a, B&& b, C&& c)
{
  log_trace("");
  auto a_base = __detail::get_ultimate_base(a);
  auto b_base = __detail::get_ultimate_base(b);

  using T = tensor_scalar_t<C>;
  using I = tensor_index_t<C>;
  using O = tensor_offset_t<C>;

  sparse_matrix_t csrA = nullptr;
  struct matrix_descr descrA;
  descrA.type = SPARSE_MATRIX_TYPE_GENERAL;
  sparse_index_base_t indexingA = SPARSE_INDEX_BASE_ZERO;
  sparse_operation_t opA = SPARSE_OPERATION_NON_TRANSPOSE;
  const index_t a_nrows = __backend::shape(a_base)[0];
  const index_t a_ncols = __backend::shape(a_base)[1];

  sparse_matrix_t csrB = nullptr;
  struct matrix_descr descrB;
  descrB.type = SPARSE_MATRIX_TYPE_GENERAL;
  sparse_index_base_t indexingB = SPARSE_INDEX_BASE_ZERO;
  sparse_operation_t opB = SPARSE_OPERATION_NON_TRANSPOSE;
  const index_t b_nrows = __backend::shape(b_base)[0];
  const index_t b_ncols = __backend::shape(b_base)[1];

  sparse_matrix_t csrC = nullptr;
  sparse_index_base_t indexingC = SPARSE_INDEX_BASE_ZERO;
  index_t c_nrows = 0, c_ncols = 0;
  offset_t c_nnz = -1;
  offset_t *c_rowptr_st = nullptr, *c_rowptr_en = nullptr;
  index_t *c_colind = nullptr;
  T *c_values = nullptr;

  __mkl_iespblas::mkl_sparse_create_csr( &csrA, indexingA, a_nrows, a_ncols, a_base.rowptr().data(),
          a_base.rowptr().data()+1, a_base.colind().data(), a_base.values().data());

  __mkl_iespblas::mkl_sparse_create_csr( &csrB, indexingB, b_nrows, b_ncols, b_base.rowptr().data(),
          b_base.rowptr().data()+1, b_base.colind().data(), b_base.values().data());

  sparse_request_t request = SPARSE_STAGE_NNZ_COUNT;
  mkl_sparse_sp2m(opA, descrA, csrA, opB, descrB, csrB, request, &csrC);

  __mkl_iespblas::mkl_sparse_export_csr( csrC, &indexingC, &c_nrows, &c_ncols, &c_rowptr_st, &c_rowptr_en,
          &c_colind, &c_values);
  offset_t c_ind = indexingC == SPARSE_INDEX_BASE_ZERO ? 0 : 1;
  c_nnz = c_rowptr_st[c_nrows] - c_ind;
  log_info("computed c_nnz = %d", c_nnz);

  return operation_info_t{
      index<>{__backend::shape(c)[0], __backend::shape(c)[1]}, c_nnz,
      __mkl_iespblas::operation_state_t{csrA, csrB, csrC}};
}



template <matrix A, matrix B, matrix C>
  requires __detail::has_csr_base<A> && __detail::has_csr_base<B> &&
           __detail::is_csr_view_v<C>
void multiply_fill(operation_info_t& info, A&& a, B&& b, C&& c)
{
  log_trace("");

  auto a_base = __detail::get_ultimate_base(a);
  auto b_base = __detail::get_ultimate_base(b);
  auto c_base = __detail::get_ultimate_base(c);

  using T = tensor_scalar_t<C>;
  using I = tensor_index_t<C>;
  using O = tensor_offset_t<C>;

  auto alpha_optional = __detail::get_scaling_factor(a, b);
  tensor_scalar_t<A> alpha = alpha_optional.value_or(1);

  sparse_matrix_t csrA = info.state_.a_handle;
  sparse_matrix_t csrB = info.state_.b_handle;
  sparse_matrix_t csrC = info.state_.c_handle;
  offset_t c_nnz = info.result_nnz();

  struct matrix_descr descrA;
  descrA.type = SPARSE_MATRIX_TYPE_GENERAL;
  sparse_index_base_t indexingA = SPARSE_INDEX_BASE_ZERO;
  sparse_operation_t opA = SPARSE_OPERATION_NON_TRANSPOSE;
  const index_t a_nrows = __backend::shape(a_base)[0];
  const index_t a_ncols = __backend::shape(a_base)[1];

  struct matrix_descr descrB;
  descrB.type = SPARSE_MATRIX_TYPE_GENERAL;
  sparse_index_base_t indexingB = SPARSE_INDEX_BASE_ZERO;
  sparse_operation_t opB = SPARSE_OPERATION_NON_TRANSPOSE;
  const index_t b_nrows = __backend::shape(b_base)[0];
  const index_t b_ncols = __backend::shape(b_base)[1];

  sparse_index_base_t indexingC = SPARSE_INDEX_BASE_ZERO;
  index_t c_nrows = __backend::shape(c_base)[0];
  index_t c_ncols = __backend::shape(c_base)[1];
  offset_t *c_rowptr_st = nullptr, *c_rowptr_en = nullptr;
  index_t *c_colind = nullptr;
  T *c_values = nullptr;

  sparse_request_t request = SPARSE_STAGE_FINALIZE_MULT;
  mkl_sparse_sp2m(opA, descrA, csrA, opB, descrB, csrB, request, &csrC);

  __mkl_iespblas::mkl_sparse_export_csr( csrC, &indexingC, &c_nrows, &c_ncols, &c_rowptr_st, &c_rowptr_en,
          &c_colind, &c_values);

  using T = tensor_scalar_t<C>;
  using I = tensor_index_t<C>;
  using O = tensor_offset_t<C>;

  // copy out of csrC arrays into C arrays
  std::copy(c_rowptr_st, c_rowptr_st+c_nrows+1, c.rowptr().begin());
  std::copy(c_colind, c_colind+c_nnz, c.colind().begin());
  std::copy(c_values, c_values+c_nnz, c.values().begin());

  if (alpha_optional.has_value()) {
    scale(alpha, c);
  }
}



} // namespace spblas

