#pragma once

#include <type_traits>

#include <cuda_runtime.h>
#include <cusparse.h>

#include <spblas/detail/operation_info_t.hpp>
#include <spblas/detail/ranges.hpp>
#include <spblas/detail/view_inspectors.hpp>

#include "types.hpp"

namespace spblas {

template <matrix A, vector B, vector C>
  requires __detail::has_csr_base<A> &&
           __detail::has_contiguous_range_base<B> &&
           __ranges::contiguous_range<C>
void multiply(A&& a, B&& b, C&& c) {
  auto a_base = __detail::get_ultimate_base(a);
  auto b_base = __detail::get_ultimate_base(b);
  using matrix_type = decltype(a_base);
  using input_type = decltype(b_base);
  using output_type = std::remove_reference_t<decltype(c)>;
  using value_type = typename matrix_type::scalar_type;

  auto alpha_optional = __detail::get_scaling_factor(a, b);
  tensor_scalar_t<A> alpha = alpha_optional.value_or(1);

  cusparseHandle_t handle = NULL;
  cusparseCreate(&handle);
  cusparseSpMatDescr_t mat;
  cusparseCreateCsr(&mat, __backend::shape(a_base)[0],
                    __backend::shape(a_base)[0], a_base.values().size(),
                    a_base.rowptr().data(), a_base.colind().data(),
                    a_base.values().data(),
                    cusparse_index_type<typename matrix_type::offset_type>(),
                    cusparse_index_type<typename matrix_type::index_type>(),
                    CUSPARSE_INDEX_BASE_ZERO, cuda_data_type<value_type>());

  cusparseDnVecDescr_t vecb, vecc;
  cusparseCreateDnVec(&vecb, b_base.size(), b_base.data(),
                      cuda_data_type<typename input_type::value_type>());
  cusparseCreateDnVec(&vecc, c.size(), c.data(),
                      cuda_data_type<typename output_type::value_type>());

  value_type alpha_val = alpha;
  value_type beta = 0.0;
  long unsigned int buffer_size = 0;
  // TODO: create a compute type for mixed precision computation
  cusparseSpMV_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha_val,
                          mat, vecb, &beta, vecc, cuda_data_type<value_type>(),
                          CUSPARSE_SPMV_ALG_DEFAULT, &buffer_size);
  void* dbuffer;
  cudaMalloc(&dbuffer, buffer_size);

  cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha_val, mat, vecb,
               &beta, vecc, cuda_data_type<value_type>(),
               CUSPARSE_SPMV_ALG_DEFAULT, dbuffer);
  cudaDeviceSynchronize();
  cusparseDestroyDnVec(vecc);
  cusparseDestroyDnVec(vecb);
  cudaFree(dbuffer);
}

template <matrix A, matrix B, matrix C>
  requires(__backend::row_iterable<A> && __backend::row_iterable<B> &&
           __detail::is_csr_view_v<C>)
void multiply(A&& a, B&& b, C&& c) {
  auto a_base = __detail::get_ultimate_base(a);
  auto b_base = __detail::get_ultimate_base(b);
  using matrix_type = decltype(a_base);
  using input_type = decltype(b_base);
  using output_type = std::remove_reference_t<decltype(c)>;
  using value_type = typename matrix_type::scalar_type;

  auto alpha_optional = __detail::get_scaling_factor(a, b);
  tensor_scalar_t<A> alpha = alpha_optional.value_or(1);

  cusparseHandle_t handle = NULL;
  cusparseSpMatDescr_t matA, matB, matC;
  void *dBuffer1 = NULL, *dBuffer2 = NULL;
  size_t bufferSize1 = 0, bufferSize2 = 0;
  typename output_type::index_type* dC_columns;
  typename output_type::scalar_type* dC_values;
  value_type alpha_val = alpha;
  value_type beta = 0.0;

  cusparseCreate(&handle); // put into info or global stuff?
  // Create sparse matrix A in CSR format
  cusparseCreateCsr(&matA, __backend::shape(a_base)[0],
                    __backend::shape(a_base)[1], a_base.values().size(),
                    a_base.rowptr().data(), a_base.colind().data(),
                    a_base.values().data(),
                    cusparse_index_type<typename matrix_type::offset_type>(),
                    cusparse_index_type<typename matrix_type::index_type>(),
                    CUSPARSE_INDEX_BASE_ZERO, cuda_data_type<value_type>());
  cusparseCreateCsr(&matB, __backend::shape(b_base)[0],
                    __backend::shape(b_base)[1], b_base.values().size(),
                    b_base.rowptr().data(), b_base.colind().data(),
                    b_base.values().data(),
                    cusparse_index_type<typename input_type::offset_type>(),
                    cusparse_index_type<typename input_type::index_type>(),
                    CUSPARSE_INDEX_BASE_ZERO,
                    cuda_data_type<typename input_type::scalar_type>());
  cusparseCreateCsr(&matC, __backend::shape(a_base)[0],
                    __backend::shape(b_base)[1], 0, c.rowptr().data(), NULL,
                    NULL,
                    cusparse_index_type<typename output_type::offset_type>(),
                    cusparse_index_type<typename output_type::index_type>(),
                    CUSPARSE_INDEX_BASE_ZERO,
                    cuda_data_type<typename output_type::scalar_type>());
  //--------------------------------------------------------------------------
  // SpGEMM Computation
  cusparseSpGEMMDescr_t spgemmDesc;
  cusparseSpGEMM_createDescr(&spgemmDesc);

  auto compute_type = cuda_data_type<typename matrix_type::scalar_type>();
  // ask bufferSize1 bytes for external memory
  cusparseSpGEMM_workEstimation(
      handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
      CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, matB, &beta, matC,
      compute_type, CUSPARSE_SPGEMM_DEFAULT, spgemmDesc, &bufferSize1, NULL);
  cudaMalloc((void**) &dBuffer1, bufferSize1);
  // inspect the matrices A and B to understand the memory requirement for
  // the next step

  cusparseSpGEMM_workEstimation(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA,
                                matB, &beta, matC, compute_type,
                                CUSPARSE_SPGEMM_DEFAULT, spgemmDesc,
                                &bufferSize1, dBuffer1);

  // ask bufferSize2 bytes for external memory

  cusparseSpGEMM_compute(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                         CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, matB,
                         &beta, matC, compute_type, CUSPARSE_SPGEMM_DEFAULT,
                         spgemmDesc, &bufferSize2, NULL);
  cudaMalloc((void**) &dBuffer2, bufferSize2);

  // compute the intermediate product of A * B
  cusparseSpGEMM_compute(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                         CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, matB,
                         &beta, matC, compute_type, CUSPARSE_SPGEMM_DEFAULT,
                         spgemmDesc, &bufferSize2, dBuffer2);
  // get matrix C non-zero entries C_nnz1
  int64_t C_num_rows1, C_num_cols1, C_nnz1;
  cusparseSpMatGetSize(matC, &C_num_rows1, &C_num_cols1, &C_nnz1);
  // allocate matrix C
  cudaMalloc((void**) &dC_columns,
             C_nnz1 * sizeof(typename output_type::index_type));
  cudaMalloc((void**) &dC_values,
             C_nnz1 * sizeof(typename output_type::scalar_type));

  // NOTE: if 'beta' != 0, the values of C must be update after the allocation
  //       of dC_values, and before the call of cusparseSpGEMM_copy

  // update matC with the new pointers

  cusparseCsrSetPointers(matC, c.rowptr().data(), dC_columns, dC_values);

  // if beta != 0, cusparseSpGEMM_copy reuses/updates the values of dC_values

  // copy the final products to the matrix C

  cusparseSpGEMM_copy(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                      CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, matB,
                      &beta, matC, compute_type, CUSPARSE_SPGEMM_DEFAULT,
                      spgemmDesc);

  std::span<typename output_type::index_type> c_colind(dC_columns, C_nnz1);
  std::span<typename output_type::scalar_type> c_values(dC_values, C_nnz1);
  c.update(c_values, c.rowptr(), c_colind,
           index<typename output_type::index_type>(__backend::shape(a)[0],
                                                   __backend::shape(b)[1]),
           C_nnz1);
  // destroy matrix/vector descriptors
  cudaDeviceSynchronize();
  cusparseSpGEMM_destroyDescr(spgemmDesc);
  cusparseDestroySpMat(matA);
  cusparseDestroySpMat(matB);
  cusparseDestroySpMat(matC);
  cusparseDestroy(handle);
  cudaFree(dBuffer1);
  cudaFree(dBuffer2);
}

// multiply_prepare() to get the buffer size

template <matrix A, matrix B, matrix C>
  requires __detail::has_csr_base<A> && __detail::has_csr_base<B> &&
           __detail::is_csr_view_v<C>
operation_info_t multiply_inspect(A&& a, B&& b, C&& c) {
  auto a_base = __detail::get_ultimate_base(a);
  auto b_base = __detail::get_ultimate_base(b);
  using matrix_type = decltype(a_base);
  using input_type = decltype(b_base);
  using output_type = std::remove_reference_t<decltype(c)>;
  using value_type = typename matrix_type::scalar_type;

  cusparseHandle_t handle = NULL;
  cusparseSpMatDescr_t matA, matB, matC;
  void *dBuffer1 = NULL, *dBuffer2 = NULL;
  size_t bufferSize1 = 0, bufferSize2 = 0;
  value_type alpha = 1.0;
  value_type beta = 0.0;
  // cudaMalloc(&dC_csrOffsets, sizeof(int) * (__backend::shape(a)[0] + 1));
  cusparseCreate(&handle);
  // Create sparse matrix A in CSR format
  cusparseCreateCsr(&matA, __backend::shape(a_base)[0],
                    __backend::shape(a_base)[1], a_base.values().size(),
                    a_base.rowptr().data(), a_base.colind().data(),
                    a_base.values().data(),
                    cusparse_index_type<typename matrix_type::offset_type>(),
                    cusparse_index_type<typename matrix_type::index_type>(),
                    CUSPARSE_INDEX_BASE_ZERO, cuda_data_type<value_type>());
  cusparseCreateCsr(&matB, __backend::shape(b_base)[0],
                    __backend::shape(b_base)[1], b_base.values().size(),
                    b_base.rowptr().data(), b_base.colind().data(),
                    b_base.values().data(),
                    cusparse_index_type<typename input_type::offset_type>(),
                    cusparse_index_type<typename input_type::index_type>(),
                    CUSPARSE_INDEX_BASE_ZERO,
                    cuda_data_type<typename input_type::scalar_type>());
  cusparseCreateCsr(&matC, __backend::shape(a_base)[0],
                    __backend::shape(b_base)[1], 0, c.rowptr().data(), NULL,
                    NULL,
                    cusparse_index_type<typename output_type::offset_type>(),
                    cusparse_index_type<typename output_type::index_type>(),
                    CUSPARSE_INDEX_BASE_ZERO,
                    cuda_data_type<typename output_type::scalar_type>());
  //--------------------------------------------------------------------------
  // SpGEMM Computation
  cusparseSpGEMMDescr_t spgemmDesc;
  cusparseSpGEMM_createDescr(&spgemmDesc);

  // ask bufferSize1 bytes for external memory
  cusparseSpGEMM_workEstimation(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA,
                                matB, &beta, matC, cuda_data_type<value_type>(),
                                CUSPARSE_SPGEMM_DEFAULT, spgemmDesc,
                                &bufferSize1, NULL);
  cudaMalloc((void**) &dBuffer1, bufferSize1);
  // inspect the matrices A and B to understand the memory requirement for
  // the next step

  cusparseSpGEMM_workEstimation(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA,
                                matB, &beta, matC, cuda_data_type<value_type>(),
                                CUSPARSE_SPGEMM_DEFAULT, spgemmDesc,
                                &bufferSize1, dBuffer1);

  // ask bufferSize2 bytes for external memory

  cusparseSpGEMM_compute(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                         CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, matB,
                         &beta, matC, cuda_data_type<value_type>(),
                         CUSPARSE_SPGEMM_DEFAULT, spgemmDesc, &bufferSize2,
                         NULL);
  cudaMalloc((void**) &dBuffer2, bufferSize2);

  // compute the intermediate product of A * B
  cusparseSpGEMM_compute(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                         CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, matB,
                         &beta, matC, cuda_data_type<value_type>(),
                         CUSPARSE_SPGEMM_DEFAULT, spgemmDesc, &bufferSize2,
                         dBuffer2);
  // get matrix C non-zero entries C_nnz1
  int64_t C_num_rows1, C_num_cols1, C_nnz1;
  cusparseSpMatGetSize(matC, &C_num_rows1, &C_num_cols1, &C_nnz1);

  //  new operation_info_t for nvidia
  auto info = operation_info_t{__backend::shape(c), C_nnz1};
  info.state.spgemm_descr = std::move(spgemmDesc);
  cusparseDestroySpMat(matA);
  cusparseDestroySpMat(matB);
  cusparseDestroySpMat(matC);
  cusparseDestroy(handle);
  cudaFree(dBuffer1);
  cudaFree(dBuffer2);
  return info;
}

template <matrix A, matrix B, matrix C>
  requires __detail::has_csr_base<A> && __detail::has_csr_base<B> &&
           __detail::is_csr_view_v<C>
void multiply_execute(operation_info_t& info, A&& a, B&& b, C&& c) {
  auto a_base = __detail::get_ultimate_base(a);
  auto b_base = __detail::get_ultimate_base(b);
  using matrix_type = decltype(a_base);
  using input_type = decltype(b_base);
  using output_type = std::remove_reference_t<decltype(c)>;
  using value_type = typename matrix_type::scalar_type;

  auto alpha_optional = __detail::get_scaling_factor(a, b);
  tensor_scalar_t<A> alpha = alpha_optional.value_or(1);

  cusparseHandle_t handle = NULL;
  cusparseSpMatDescr_t matA, matB, matC;
  value_type alpha_val = alpha;
  value_type beta = 0.0;
  // cudaMalloc(&dC_csrOffsets, sizeof(int) * (__backend::shape(a)[0] + 1));
  cusparseCreate(&handle);
  // Create sparse matrix A in CSR format
  cusparseCreateCsr(&matA, __backend::shape(a_base)[0],
                    __backend::shape(a_base)[1], a_base.values().size(),
                    a_base.rowptr().data(), a_base.colind().data(),
                    a_base.values().data(),
                    cusparse_index_type<typename matrix_type::offset_type>(),
                    cusparse_index_type<typename matrix_type::index_type>(),
                    CUSPARSE_INDEX_BASE_ZERO, cuda_data_type<value_type>());
  cusparseCreateCsr(&matB, __backend::shape(b_base)[0],
                    __backend::shape(b_base)[1], b_base.values().size(),
                    b_base.rowptr().data(), b_base.colind().data(),
                    b_base.values().data(),
                    cusparse_index_type<typename input_type::offset_type>(),
                    cusparse_index_type<typename input_type::index_type>(),
                    CUSPARSE_INDEX_BASE_ZERO,
                    cuda_data_type<typename input_type::scalar_type>());
  cusparseCreateCsr(&matC, __backend::shape(c)[0], __backend::shape(c)[1],
                    c.values().size(), c.rowptr().data(), c.colind().data(),
                    c.values().data(),
                    cusparse_index_type<typename output_type::offset_type>(),
                    cusparse_index_type<typename output_type::index_type>(),
                    CUSPARSE_INDEX_BASE_ZERO,
                    cuda_data_type<typename output_type::scalar_type>());

  cusparseSpGEMM_copy(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                      CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, matB,
                      &beta, matC, cuda_data_type<value_type>(),
                      CUSPARSE_SPGEMM_DEFAULT, info.state.spgemm_descr);
  // destroy matrix/vector descriptors
  cudaDeviceSynchronize();
  cusparseDestroySpMat(matA);
  cusparseDestroySpMat(matB);
  cusparseDestroySpMat(matC);
  cusparseDestroy(handle);
}

} // namespace spblas
