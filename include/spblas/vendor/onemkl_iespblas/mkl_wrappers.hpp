#pragma once

#include <cstdint>
#include "mkl.h"

// 
// Add several templated functions for mapping from data_type to C style IE Sparse BLAS APIs
//


namespace spblas {
namespace __mkl_iespblas {

//
// mkl_sparse_create_csr
//
template<class T>
inline sparse_status_t mkl_sparse_create_csr( sparse_matrix_t *csrA, const sparse_index_base_t indexing,
                       const MKL_INT nrows, const MKL_INT ncols, MKL_INT *rowptr_st,
                       MKL_INT *rowptr_en, MKL_INT *colind, T *values)
{ 
  std::cout << "mkl_sparse_create_csr data types are not supported" << std::endl;
  return SPARSE_STATUS_NOT_SUPPORTED;
}

template<>
inline sparse_status_t mkl_sparse_create_csr<float>( sparse_matrix_t *csrA, const sparse_index_base_t indexing,
                       const MKL_INT nrows, const MKL_INT ncols, MKL_INT *rowptr_st,
                       MKL_INT *rowptr_en, MKL_INT *colind, float *values)
{
    return mkl_sparse_s_create_csr(csrA, indexing, nrows, ncols, rowptr_st, rowptr_en, colind, values);
}

template<>
inline sparse_status_t mkl_sparse_create_csr<double>( sparse_matrix_t *csrA, const sparse_index_base_t indexing,
                      const MKL_INT nrows, const MKL_INT ncols, MKL_INT *rowptr_st,
                      MKL_INT *rowptr_en, MKL_INT *colind, double *values)
{
    return mkl_sparse_d_create_csr(csrA, indexing, nrows, ncols, rowptr_st, rowptr_en, colind, values);
}


//
// mkl_sparse_export_csr
//

template<class T>
inline sparse_status_t mkl_sparse_export_csr( const sparse_matrix_t csrA, sparse_index_base_t *indexing,
                       MKL_INT *nrows, MKL_INT *ncols, MKL_INT **rowptr_st,
                       MKL_INT **rowptr_en, MKL_INT **colind, T **values)
{ 
  std::cout << "mkl_sparse_export_csr data types are not supported" << std::endl;
  return SPARSE_STATUS_NOT_SUPPORTED;
}

template<>
inline sparse_status_t mkl_sparse_export_csr<float>( const sparse_matrix_t csrA, sparse_index_base_t *indexing,
                       MKL_INT *nrows, MKL_INT *ncols, MKL_INT **rowptr_st,
                       MKL_INT **rowptr_en, MKL_INT **colind, float **values)
{
    return mkl_sparse_s_export_csr(csrA, indexing, nrows, ncols, rowptr_st, rowptr_en, colind, values);
}

template<>
inline sparse_status_t mkl_sparse_export_csr<double>( const sparse_matrix_t csrA, sparse_index_base_t *indexing,
                       MKL_INT *nrows, MKL_INT *ncols, MKL_INT **rowptr_st,
                       MKL_INT **rowptr_en, MKL_INT **colind, double **values)
{
    return mkl_sparse_d_export_csr(csrA, indexing, nrows, ncols, rowptr_st, rowptr_en, colind, values);
}


//
// mkl_sparse_mv
//
template<class T>
inline sparse_status_t mkl_sparse_mv( const sparse_operation_t op, const T alpha, const sparse_matrix_t csrA,
                               const struct matrix_descr descr, const T* x, const T beta, T* y)
{ 
  std::cout << "mkl_sparse_mv data types are not supported" << std::endl;
  return SPARSE_STATUS_NOT_SUPPORTED;
}

template<>
inline sparse_status_t mkl_sparse_mv<float>( const sparse_operation_t op, const float alpha, const sparse_matrix_t csrA,
                                      const struct matrix_descr descr, const float* x, const float beta, float* y)
{
    return mkl_sparse_s_mv(op, alpha, csrA, descr, x, beta, y);
}

template<>
inline sparse_status_t mkl_sparse_mv<double>( const sparse_operation_t op, const double alpha, const sparse_matrix_t csrA,
                                       const struct matrix_descr descr, const double* x, const double beta, double* y)
{
    return mkl_sparse_d_mv(op, alpha, csrA, descr, x, beta, y);
}


//
// mkl_sparse_mm
//
template<class T>
inline sparse_status_t mkl_sparse_mm( const sparse_operation_t op, const T alpha, const sparse_matrix_t csrA,
                               const struct matrix_descr descr, const sparse_layout_t layout, 
                               const T* x, const index_t nrhs, const index_t ldx, const T beta, T* y, const index_t ldy)
{ 
  std::cout << "mkl_sparse_mm data types are not supported" << std::endl;
  return SPARSE_STATUS_NOT_SUPPORTED;
}

template<>
inline sparse_status_t mkl_sparse_mm<float>( const sparse_operation_t op, const float alpha, const sparse_matrix_t csrA,
                                      const struct matrix_descr descr, const sparse_layout_t layout,
                                      const float* x, const index_t nrhs, const index_t ldx, const float beta, float* y, const index_t ldy)
{
    return mkl_sparse_s_mm(op, alpha, csrA, descr, layout, x, nrhs, ldx, beta, y, ldy);
}

template<>
inline sparse_status_t mkl_sparse_mm<double>( const sparse_operation_t op, const double alpha, const sparse_matrix_t csrA,
                                       const struct matrix_descr descr, const sparse_layout_t layout,
                                       const double* x, const index_t nrhs, const index_t ldx, const double beta, double* y, const index_t ldy)
{
    return mkl_sparse_d_mm(op, alpha, csrA, descr, layout, x, nrhs, ldx, beta, y, ldy);
}



} // namespace __mkl_iespblas
} // namespace spblas




