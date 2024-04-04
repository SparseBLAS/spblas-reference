#pragma once

#include <spblas/backend/backend.hpp>
#include <spblas/concepts.hpp>

#include <spblas/backend/csr_builder.hpp>
#include <spblas/backend/spa_accumulator.hpp>
#include <spblas/detail/operation_info_t.hpp>

#include <algorithm>

// #include <cuda.h>
#include <cuda_runtime.h>
#include <cusparse.h>

namespace spblas {

// C = AB
// SpMV
template <matrix A, vector B, vector C>
void multiply(A&& a, B&& b, C&& c) {
  if (__backend::shape(a)[0] != __backend::shape(c) ||
      __backend::shape(a)[1] != __backend::shape(b)) {
    throw std::invalid_argument(
        "multiply: matrix and vector dimensions are incompatible.");
  }

  // dispatch the matrix type by concept?
  // TODO: it should be an helper function to get/identify scalar, transpose,
  // conjugate, and view
  cusparseHandle_t handle = NULL;
  cusparseCreate(&handle);
  cusparseSpMatDescr_t mat;
  cusparseCreateCsr(&mat, __backend::shape(a)[0], __backend::shape(a)[0],
                    a.values().size(), a.rowptr().data(), a.colind().data(),
                    a.values().data(), CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                    CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);

  cusparseDnVecDescr_t vecb, vecc;
  cusparseCreateDnVec(&vecb, b.size(), b.data(), CUDA_R_32F);
  cusparseCreateDnVec(&vecc, c.size(), c.data(), CUDA_R_32F);

  float alpha = 1.0;
  float beta = 0.0;
  long unsigned int buffer_size = 0;
  cusparseSpMV_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, mat,
                          vecb, &beta, vecc, CUDA_R_32F,
                          CUSPARSE_SPMV_ALG_DEFAULT, &buffer_size);
  void* dbuffer;
  cudaMalloc(&dbuffer, buffer_size);

  cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, mat, vecb,
               &beta, vecc, CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, dbuffer);
  cudaDeviceSynchronize();
  cusparseDestroyDnVec(vecc);
  cusparseDestroyDnVec(vecb);
  cudaFree(dbuffer);
}

// C = AB
// SpMM
template <matrix A, matrix B, matrix C>
  requires(__backend::lookupable<B> && __backend::lookupable<C>)
void multiply(A&& a, B&& b, C&& c) {
  if (__backend::shape(a)[0] != __backend::shape(c)[0] ||
      __backend::shape(b)[1] != __backend::shape(c)[1] ||
      __backend::shape(a)[1] != __backend::shape(b)[0]) {
    throw std::invalid_argument(
        "multiply: matrix dimensions are incompatible.");
  }
}

// C = AB
// SpGEMM (Gustavson's Algorithm;
template <matrix A, matrix B, matrix C>
  requires(__backend::row_iterable<A> && __backend::row_iterable<B> &&
           __detail::is_csr_view_v<C>)
void multiply(A&& a, B&& b, C&& c) {
  if (__backend::shape(a)[0] != __backend::shape(c)[0] ||
      __backend::shape(b)[1] != __backend::shape(c)[1] ||
      __backend::shape(a)[1] != __backend::shape(b)[0]) {
    throw std::invalid_argument(
        "multiply: matrix dimensions are incompatible.");
  }

  cusparseHandle_t handle = NULL;
  cusparseSpMatDescr_t matA, matB, matC;
  void *dBuffer1 = NULL, *dBuffer2 = NULL;
  size_t bufferSize1 = 0, bufferSize2 = 0;
  int* dC_columns;
  float* dC_values;
  float alpha = 1.0;
  float beta = 0.0;
  // cudaMalloc(&dC_csrOffsets, sizeof(int) * (__backend::shape(a)[0] + 1));
  cusparseCreate(&handle);
  // Create sparse matrix A in CSR format
  cusparseCreateCsr(&matA, __backend::shape(a)[0], __backend::shape(a)[1],
                    a.values().size(), a.rowptr().data(), a.colind().data(),
                    a.values().data(), CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                    CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);
  cusparseCreateCsr(&matB, __backend::shape(b)[0], __backend::shape(b)[1],
                    b.values().size(), b.rowptr().data(), b.colind().data(),
                    b.values().data(), CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                    CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);
  cusparseCreateCsr(&matC, __backend::shape(a)[0], __backend::shape(b)[1], 0,
                    c.rowptr().data(), NULL, NULL, CUSPARSE_INDEX_32I,
                    CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);
  //--------------------------------------------------------------------------
  // SpGEMM Computation
  cusparseSpGEMMDescr_t spgemmDesc;
  cusparseSpGEMM_createDescr(&spgemmDesc);

  // ask bufferSize1 bytes for external memory

  cusparseSpGEMM_workEstimation(
      handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
      CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, matB, &beta, matC,
      CUDA_R_32F, CUSPARSE_SPGEMM_DEFAULT, spgemmDesc, &bufferSize1, NULL);
  cudaMalloc((void**)&dBuffer1, bufferSize1);
  // inspect the matrices A and B to understand the memory requirement for
  // the next step

  cusparseSpGEMM_workEstimation(
      handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
      CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, matB, &beta, matC,
      CUDA_R_32F, CUSPARSE_SPGEMM_DEFAULT, spgemmDesc, &bufferSize1, dBuffer1);

  // ask bufferSize2 bytes for external memory

  cusparseSpGEMM_compute(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                         CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, matB,
                         &beta, matC, CUDA_R_32F, CUSPARSE_SPGEMM_DEFAULT,
                         spgemmDesc, &bufferSize2, NULL);
  cudaMalloc((void**)&dBuffer2, bufferSize2);

  // compute the intermediate product of A * B
  cusparseSpGEMM_compute(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                         CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, matB,
                         &beta, matC, CUDA_R_32F, CUSPARSE_SPGEMM_DEFAULT,
                         spgemmDesc, &bufferSize2, dBuffer2);
  // get matrix C non-zero entries C_nnz1
  int64_t C_num_rows1, C_num_cols1, C_nnz1;
  cusparseSpMatGetSize(matC, &C_num_rows1, &C_num_cols1, &C_nnz1);
  // allocate matrix C
  cudaMalloc((void**)&dC_columns, C_nnz1 * sizeof(int));
  cudaMalloc((void**)&dC_values, C_nnz1 * sizeof(float));

  // NOTE: if 'beta' != 0, the values of C must be update after the allocation
  //       of dC_values, and before the call of cusparseSpGEMM_copy

  // update matC with the new pointers

  cusparseCsrSetPointers(matC, c.rowptr().data(), dC_columns, dC_values);

  // if beta != 0, cusparseSpGEMM_copy reuses/updates the values of dC_values

  // copy the final products to the matrix C

  cusparseSpGEMM_copy(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                      CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, matB,
                      &beta, matC, CUDA_R_32F, CUSPARSE_SPGEMM_DEFAULT,
                      spgemmDesc);

  std::span<int> c_colind(dC_columns, C_nnz1);
  std::span<float> c_values(dC_values, C_nnz1);
  c.update(c_values, c.rowptr(), c_colind,
           index<int>(__backend::shape(a)[0], __backend::shape(b)[1]), C_nnz1);
  // destroy matrix/vector descriptors
  cudaDeviceSynchronize();
  cusparseSpGEMM_destroyDescr(spgemmDesc);
  cusparseDestroySpMat(matA);
  cusparseDestroySpMat(matB);
  cusparseDestroySpMat(matC);
  cusparseDestroy(handle);
}

// C = AB
// SpGEMM (Gustavson's Algorithm;
template <matrix A, matrix B, matrix C>
  requires(__backend::row_iterable<A> && __backend::row_iterable<B> &&
           __detail::is_csr_view_v<C>)
operation_info_t multiply_inspect(A&& a, B&& b, C&& c) {
  if (__backend::shape(a)[0] != __backend::shape(c)[0] ||
      __backend::shape(b)[1] != __backend::shape(c)[1] ||
      __backend::shape(a)[1] != __backend::shape(b)[0]) {
    throw std::invalid_argument(
        "multiply: matrix dimensions are incompatible.");
  }

  cusparseHandle_t handle = NULL;
  cusparseSpMatDescr_t matA, matB, matC;
  void *dBuffer1 = NULL, *dBuffer2 = NULL;
  size_t bufferSize1 = 0, bufferSize2 = 0;
  int* dC_columns;
  float* dC_values;
  float alpha = 1.0;
  float beta = 0.0;
  // cudaMalloc(&dC_csrOffsets, sizeof(int) * (__backend::shape(a)[0] + 1));
  cusparseCreate(&handle);
  // Create sparse matrix A in CSR format
  cusparseCreateCsr(&matA, __backend::shape(a)[0], __backend::shape(a)[1],
                    a.values().size(), a.rowptr().data(), a.colind().data(),
                    a.values().data(), CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                    CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);
  cusparseCreateCsr(&matB, __backend::shape(b)[0], __backend::shape(b)[1],
                    b.values().size(), b.rowptr().data(), b.colind().data(),
                    b.values().data(), CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                    CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);
  cusparseCreateCsr(&matC, __backend::shape(a)[0], __backend::shape(b)[1], 0,
                    c.rowptr().data(), NULL, NULL, CUSPARSE_INDEX_32I,
                    CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);
  //--------------------------------------------------------------------------
  // SpGEMM Computation
  cusparseSpGEMMDescr_t spgemmDesc;
  cusparseSpGEMM_createDescr(&spgemmDesc);

  // ask bufferSize1 bytes for external memory

  cusparseSpGEMM_workEstimation(
      handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
      CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, matB, &beta, matC,
      CUDA_R_32F, CUSPARSE_SPGEMM_DEFAULT, spgemmDesc, &bufferSize1, NULL);
  cudaMalloc((void**)&dBuffer1, bufferSize1);
  // inspect the matrices A and B to understand the memory requirement for
  // the next step

  cusparseSpGEMM_workEstimation(
      handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
      CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, matB, &beta, matC,
      CUDA_R_32F, CUSPARSE_SPGEMM_DEFAULT, spgemmDesc, &bufferSize1, dBuffer1);

  // ask bufferSize2 bytes for external memory

  cusparseSpGEMM_compute(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                         CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, matB,
                         &beta, matC, CUDA_R_32F, CUSPARSE_SPGEMM_DEFAULT,
                         spgemmDesc, &bufferSize2, NULL);
  cudaMalloc((void**)&dBuffer2, bufferSize2);

  // compute the intermediate product of A * B
  cusparseSpGEMM_compute(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                         CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, matB,
                         &beta, matC, CUDA_R_32F, CUSPARSE_SPGEMM_DEFAULT,
                         spgemmDesc, &bufferSize2, dBuffer2);
  // get matrix C non-zero entries C_nnz1
  int64_t C_num_rows1, C_num_cols1, C_nnz1;
  cusparseSpMatGetSize(matC, &C_num_rows1, &C_num_cols1, &C_nnz1);

  //  new operation_info_t for nvidia
  auto info = operation_info_t{__backend::shape(c), C_nnz1};
  info.spgemm_descr = std::move(spgemmDesc);
  return info;
}

// C = AB
template <matrix A, matrix B, matrix C>
void multiply_execute(operation_info_t info, A&& a, B&& b, C&& c) {

  cusparseHandle_t handle = NULL;
  cusparseSpMatDescr_t matA, matB, matC;
  float alpha = 1.0;
  float beta = 0.0;
  // cudaMalloc(&dC_csrOffsets, sizeof(int) * (__backend::shape(a)[0] + 1));
  cusparseCreate(&handle);
  // Create sparse matrix A in CSR format
  cusparseCreateCsr(&matA, __backend::shape(a)[0], __backend::shape(a)[1],
                    a.values().size(), a.rowptr().data(), a.colind().data(),
                    a.values().data(), CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                    CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);
  cusparseCreateCsr(&matB, __backend::shape(b)[0], __backend::shape(b)[1],
                    b.values().size(), b.rowptr().data(), b.colind().data(),
                    b.values().data(), CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                    CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);
  cusparseCreateCsr(&matC, __backend::shape(c)[0], __backend::shape(c)[1],
                    c.values().size(), c.rowptr().data(), c.colind().data(),
                    c.values().data(), CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                    CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);

  cusparseSpGEMM_copy(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                      CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, matB,
                      &beta, matC, CUDA_R_32F, CUSPARSE_SPGEMM_DEFAULT,
                      info.spgemm_descr);
  // destroy matrix/vector descriptors
  cudaDeviceSynchronize();
  cusparseDestroySpMat(matA);
  cusparseDestroySpMat(matB);
  cusparseDestroySpMat(matC);
  cusparseDestroy(handle);
}

} // namespace spblas
