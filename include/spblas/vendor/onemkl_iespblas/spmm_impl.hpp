#pragma once

#include "mkl.h"

#include <spblas/backend/log.hpp>

#include "mkl_wrappers.hpp"
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
// //operation_info_t multiply_inspect(A, x, y)
// //void multiply_inspect(operation_info_t, A, x, y)
//
// //void multiply_execute(operation_info_t, A, x, y)
// void multiply(A, x, y)
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

  sparse_matrix_t csrA = nullptr;
  struct matrix_descr descr;
  descr.type = SPARSE_MATRIX_TYPE_GENERAL;
  sparse_index_base_t indexing = SPARSE_INDEX_BASE_ZERO;
  sparse_operation_t opA = SPARSE_OPERATION_NON_TRANSPOSE;
  sparse_layout_t layout = SPARSE_LAYOUT_ROW_MAJOR;

  const index_t a_nrows = __backend::shape(a_base)[0];
  const index_t a_ncols = __backend::shape(a_base)[1];
  const index_t nrhs = x_base.extent(1);
  const index_t ldx  = x_base.extent(1);
  const index_t ldy  = y_base.extent(1);

  __mkl_iespblas::mkl_sparse_create_csr( &csrA, indexing, a_nrows, a_ncols, a_base.rowptr().data(),
          a_base.rowptr().data()+1, a_base.colind().data(), a_base.values().data());

  mkl_sparse_set_mm_hint( csrA, opA, descr, layout, nrhs, 1);

  mkl_sparse_optimize( csrA );

  T beta = static_cast<T>(0.0);
  __mkl_iespblas::mkl_sparse_mm( opA, alpha, csrA, descr, layout, x_base.data_handle(),
          nrhs, ldx, beta, y.data_handle(), ldy);

  mkl_sparse_destroy( csrA );
  
}



} // namespace spblas
