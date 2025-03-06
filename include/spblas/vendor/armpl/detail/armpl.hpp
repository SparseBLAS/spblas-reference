#pragma once

#include <complex>

#define armpl_singlecomplex_t std::complex<float>
#define armpl_doublecomplex_t std::complex<double>

#include <armpl_sparse.h>

namespace spblas {

namespace __armpl {

template <typename T>
armpl_status_t (*create_spmat_csr)(armpl_spmat_t*, armpl_int_t, armpl_int_t,
                                   const armpl_int_t*, const armpl_int_t*,
                                   const T*, armpl_int_t);
template <>
inline constexpr auto create_spmat_csr<float> = &armpl_spmat_create_csr_s;
template <>
inline constexpr auto create_spmat_csr<double> = &armpl_spmat_create_csr_d;
template <>
inline constexpr auto create_spmat_csr<std::complex<float>> =
    &armpl_spmat_create_csr_c;
template <>
inline constexpr auto create_spmat_csr<std::complex<double>> =
    &armpl_spmat_create_csr_z;

template <typename T>
armpl_status_t (*create_spmat_csc)(armpl_spmat_t*, armpl_int_t, armpl_int_t,
                                   const armpl_int_t*, const armpl_int_t*,
                                   const T*, armpl_int_t);
template <>
inline constexpr auto create_spmat_csc<float> = &armpl_spmat_create_csc_s;
template <>
inline constexpr auto create_spmat_csc<double> = &armpl_spmat_create_csc_d;
template <>
inline constexpr auto create_spmat_csc<std::complex<float>> =
    &armpl_spmat_create_csc_c;
template <>
inline constexpr auto create_spmat_csc<std::complex<double>> =
    &armpl_spmat_create_csc_z;

template <typename T>
armpl_status_t (*create_spmat_dense)(armpl_spmat_t*, enum armpl_dense_layout,
                                     armpl_int_t, armpl_int_t, armpl_int_t,
                                     const T*, armpl_int_t);
template <>
inline constexpr auto create_spmat_dense<float> = &armpl_spmat_create_dense_s;
template <>
inline constexpr auto create_spmat_dense<double> = &armpl_spmat_create_dense_d;
template <>
inline constexpr auto create_spmat_dense<std::complex<float>> =
    &armpl_spmat_create_dense_c;
template <>
inline constexpr auto create_spmat_dense<std::complex<double>> =
    &armpl_spmat_create_dense_z;

template <typename T>
armpl_status_t (*spmv_exec)(enum armpl_sparse_hint_value, T, armpl_spmat_t,
                            const T*, T, T*);
template <>
inline constexpr auto spmv_exec<float> = &armpl_spmv_exec_s;
template <>
inline constexpr auto spmv_exec<double> = &armpl_spmv_exec_d;
template <>
inline constexpr auto spmv_exec<std::complex<float>> = &armpl_spmv_exec_c;
template <>
inline constexpr auto spmv_exec<std::complex<double>> = &armpl_spmv_exec_z;

template <typename T>
armpl_status_t (*spmm_exec)(enum armpl_sparse_hint_value,
                            enum armpl_sparse_hint_value, T, armpl_spmat_t,
                            armpl_spmat_t, T, armpl_spmat_t);
template <>
inline constexpr auto spmm_exec<float> = &armpl_spmm_exec_s;
template <>
inline constexpr auto spmm_exec<double> = &armpl_spmm_exec_d;
template <>
inline constexpr auto spmm_exec<std::complex<float>> = &armpl_spmm_exec_c;
template <>
inline constexpr auto spmm_exec<std::complex<double>> = &armpl_spmm_exec_z;

template <typename T>
armpl_status_t (*sptrsv_exec)(enum armpl_sparse_hint_value, armpl_spmat_t, T*,
                              T, const T*);
template <>
inline constexpr auto sptrsv_exec<float> = &armpl_spsv_exec_s;
template <>
inline constexpr auto sptrsv_exec<double> = &armpl_spsv_exec_d;
template <>
inline constexpr auto sptrsv_exec<std::complex<float>> = &armpl_spsv_exec_c;
template <>
inline constexpr auto sptrsv_exec<std::complex<double>> = &armpl_spsv_exec_z;

template <typename T>
armpl_status_t (*export_spmat_dense)(armpl_const_spmat_t,
                                     enum armpl_dense_layout, armpl_int_t*,
                                     armpl_int_t*, const T**);
template <>
inline constexpr auto export_spmat_dense<float> = &armpl_spmat_export_dense_s;
template <>
inline constexpr auto export_spmat_dense<double> = &armpl_spmat_export_dense_d;
template <>
inline constexpr auto export_spmat_dense<std::complex<float>> =
    &armpl_spmat_export_dense_c;
template <>
inline constexpr auto export_spmat_dense<std::complex<double>> =
    &armpl_spmat_export_dense_z;

template <typename T>
armpl_status_t (*export_spmat_csr)(armpl_const_spmat_t, armpl_int_t,
                                   armpl_int_t*, armpl_int_t*,
                                   const armpl_int_t**, const armpl_int_t**,
                                   const T**);
template <>
inline constexpr auto export_spmat_csr<float> = &armpl_spmat_export_csr_s;
template <>
inline constexpr auto export_spmat_csr<double> = &armpl_spmat_export_csr_d;
template <>
inline constexpr auto export_spmat_csr<std::complex<float>> =
    &armpl_spmat_export_csr_c;
template <>
inline constexpr auto export_spmat_csr<std::complex<double>> =
    &armpl_spmat_export_csr_z;

} // namespace __armpl

} // namespace spblas
