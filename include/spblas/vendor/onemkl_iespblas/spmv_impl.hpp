#pragma once

#include "mkl.h"

#include <spblas/backend/log.hpp>

#include "mkl_wrappers.hpp"
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
//
// //operation_info_t multiply_inspect(A, x, y)
// //void multiply_inspect(operation_info_t, A, x, y)
//
// //void multiply_execute(operation_info_t, A, x, y)
// void multiply(A, x, y)
//


namespace spblas {

template <matrix A, vector X, vector Y>
  requires __detail::has_csr_base<A> &&
           __detail::has_contiguous_range_base<X> &&
           __ranges::contiguous_range<Y>
void multiply(A&& a, X&& x, Y&& y)
{
  log_trace("");
  auto a_base = __detail::get_ultimate_base(a);
  auto x_base = __detail::get_ultimate_base(x);
  
  using T = tensor_scalar_t<A>;
  using I = tensor_index_t<A>;
  using O = tensor_offset_t<A>;

  auto alpha_optional = __detail::get_scaling_factor(a, x);
  T alpha = alpha_optional.value_or(1);

  sparse_matrix_t csrA = nullptr;
  struct matrix_descr descr;
  descr.type = SPARSE_MATRIX_TYPE_GENERAL;
  sparse_index_base_t indexing = SPARSE_INDEX_BASE_ZERO;
  sparse_operation_t opA = SPARSE_OPERATION_NON_TRANSPOSE;

  const index_t a_nrows = __backend::shape(a_base)[0];
  const index_t a_ncols = __backend::shape(a_base)[1];

  __mkl_iespblas::mkl_sparse_create_csr( &csrA, indexing, a_nrows, a_ncols, a_base.rowptr().data(),
          a_base.rowptr().data()+1, a_base.colind().data(), a_base.values().data());

  mkl_sparse_set_mv_hint( csrA, opA, descr, 1);

  mkl_sparse_optimize( csrA );

  T beta = static_cast<T>(0.0);
  __mkl_iespblas::mkl_sparse_mv( opA, alpha, csrA, descr, __ranges::data(x_base), beta, __ranges::data(y));

  mkl_sparse_destroy( csrA );

}


} // namespace spblas
