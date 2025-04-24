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
#include <spblas/detail/log.hpp>

namespace spblas {
namespace __aoclsparse {

template <typename T>
aoclsparse_status
aoclsparse_csrmm(aoclsparse_operation op, T alpha, const aoclsparse_matrix A,
                 const aoclsparse_mat_descr descr, aoclsparse_order order,
                 const T* B, aoclsparse_int n, aoclsparse_int ldb, T beta, T* C,
                 aoclsparse_int ldc) {
  log_warning("spmm data types are currently not supported");
  return aoclsparse_status_not_implemented;
}

template <>
inline aoclsparse_status aoclsparse_csrmm<float>(
    aoclsparse_operation op, float alpha, const aoclsparse_matrix A,
    const aoclsparse_mat_descr descr, aoclsparse_order order, const float* B,
    aoclsparse_int n, aoclsparse_int ldb, float beta, float* C,
    aoclsparse_int ldc) {
  return aoclsparse_scsrmm(op, alpha, A, descr, order, B, n, ldb, beta, C, ldc);
}

template <>
inline aoclsparse_status aoclsparse_csrmm<double>(
    aoclsparse_operation op, double alpha, const aoclsparse_matrix A,
    const aoclsparse_mat_descr descr, aoclsparse_order order, const double* B,
    aoclsparse_int n, aoclsparse_int ldb, double beta, double* C,
    aoclsparse_int ldc) {
  return aoclsparse_dcsrmm(op, alpha, A, descr, order, B, n, ldb, beta, C, ldc);
}

template <typename T>
inline aoclsparse_status aoclsparse_mv(aoclsparse_operation op, const T* alpha,
                                       aoclsparse_matrix A,
                                       const aoclsparse_mat_descr descr,
                                       const T* x, const T* beta, T* y) {
  log_warning("aoclsparse_mv data types are currently not supported");
  return aoclsparse_status_not_implemented;
}

template <>
inline aoclsparse_status
aoclsparse_mv<float>(aoclsparse_operation op, const float* alpha,
                     aoclsparse_matrix A, const aoclsparse_mat_descr descr,
                     const float* x, const float* beta, float* y) {
  return aoclsparse_smv(op, alpha, A, descr, x, beta, y);
}

template <>
inline aoclsparse_status
aoclsparse_mv<double>(aoclsparse_operation op, const double* alpha,
                      aoclsparse_matrix A, const aoclsparse_mat_descr descr,
                      const double* x, const double* beta, double* y) {
  return aoclsparse_dmv(op, alpha, A, descr, x, beta, y);
}

template <typename T>
inline aoclsparse_status aoclsparse_trsv(const aoclsparse_operation trans,
                                         const T alpha, aoclsparse_matrix A,
                                         const aoclsparse_mat_descr descr,
                                         const T* b, T* x) {
  log_warning("aoclsparse_trsv data types are currently not supported");
  return aoclsparse_status_not_implemented;
}

template <>
inline aoclsparse_status
aoclsparse_trsv<double>(const aoclsparse_operation trans, const double alpha,
                        aoclsparse_matrix A, const aoclsparse_mat_descr descr,
                        const double* b, double* x) {
  return aoclsparse_dtrsv(trans, alpha, A, descr, b, x);
}

template <>
inline aoclsparse_status
aoclsparse_trsv<float>(const aoclsparse_operation trans, const float alpha,
                       aoclsparse_matrix A, const aoclsparse_mat_descr descr,
                       const float* b, float* x) {
  return aoclsparse_strsv(trans, alpha, A, descr, b, x);
}

template <class T>
inline aoclsparse_status
aoclsparse_create_csr(aoclsparse_matrix* mat, aoclsparse_index_base base,
                      aoclsparse_int M, aoclsparse_int N, aoclsparse_int nnz,
                      aoclsparse_int* row_ptr, aoclsparse_int* col_idx,
                      T* val) {
  log_warning("matrix creating with this data type is currently not supported");
  return aoclsparse_status_not_implemented;
}

template <>
inline aoclsparse_status
aoclsparse_create_csr<float>(aoclsparse_matrix* mat, aoclsparse_index_base base,
                             aoclsparse_int M, aoclsparse_int N,
                             aoclsparse_int nnz, aoclsparse_int* row_ptr,
                             aoclsparse_int* col_idx, float* val) {
  return aoclsparse_create_scsr(mat, base, M, N, nnz, row_ptr, col_idx, val);
}
template <>
inline aoclsparse_status aoclsparse_create_csr<double>(
    aoclsparse_matrix* mat, aoclsparse_index_base base, aoclsparse_int M,
    aoclsparse_int N, aoclsparse_int nnz, aoclsparse_int* row_ptr,
    aoclsparse_int* col_idx, double* val) {
  return aoclsparse_create_dcsr(mat, base, M, N, nnz, row_ptr, col_idx, val);
}

template <class T>
inline aoclsparse_status
aoclsparse_create_csc(aoclsparse_matrix* mat, aoclsparse_index_base base,
                      aoclsparse_int M, aoclsparse_int N, aoclsparse_int nnz,
                      aoclsparse_int* col_ptr, aoclsparse_int* row_idx,
                      T* val) {
  log_warning("matrix creating with this data type is currently not supported");
  return aoclsparse_status_not_implemented;
}

template <>
inline aoclsparse_status
aoclsparse_create_csc<float>(aoclsparse_matrix* mat, aoclsparse_index_base base,
                             aoclsparse_int M, aoclsparse_int N,
                             aoclsparse_int nnz, aoclsparse_int* col_ptr,
                             aoclsparse_int* row_idx, float* val) {
  return aoclsparse_create_scsc(mat, base, M, N, nnz, col_ptr, row_idx, val);
}
template <>
inline aoclsparse_status aoclsparse_create_csc<double>(
    aoclsparse_matrix* mat, aoclsparse_index_base base, aoclsparse_int M,
    aoclsparse_int N, aoclsparse_int nnz, aoclsparse_int* col_ptr,
    aoclsparse_int* row_idx, double* val) {
  return aoclsparse_create_dcsc(mat, base, M, N, nnz, col_ptr, row_idx, val);
}

template <typename T>
inline aoclsparse_status
aoclsparse_export_csr(const aoclsparse_matrix mat, aoclsparse_index_base* base,
                      aoclsparse_int* m, aoclsparse_int* n, aoclsparse_int* nnz,
                      aoclsparse_int** row_ptr, aoclsparse_int** col_idx,
                      T** val) {
  log_warning(
      "exporting matrix with this data type is currently not supported");
  return aoclsparse_status_not_implemented;
}

template <>
inline aoclsparse_status aoclsparse_export_csr<float>(
    const aoclsparse_matrix mat, aoclsparse_index_base* base, aoclsparse_int* m,
    aoclsparse_int* n, aoclsparse_int* nnz, aoclsparse_int** row_ptr,
    aoclsparse_int** col_idx, float** val) {
  return aoclsparse_export_scsr(mat, base, m, n, nnz, row_ptr, col_idx, val);
}

template <>
inline aoclsparse_status aoclsparse_export_csr<double>(
    const aoclsparse_matrix mat, aoclsparse_index_base* base, aoclsparse_int* m,
    aoclsparse_int* n, aoclsparse_int* nnz, aoclsparse_int** row_ptr,
    aoclsparse_int** col_idx, double** val) {
  return aoclsparse_export_dcsr(mat, base, m, n, nnz, row_ptr, col_idx, val);
}

} // namespace __aoclsparse
} // namespace spblas
