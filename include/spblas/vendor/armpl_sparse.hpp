#pragma once

#include <armpl_sparse.h>
#include <complex>

namespace armpl {

template<class T>
armpl_status_t (*create_spmat_csr)(armpl_spmat_t *, armpl_int_t, armpl_int_t, const armpl_int_t *, const armpl_int_t *, const T *, armpl_int_t);
template<> inline constexpr auto create_spmat_csr<float> = &armpl_spmat_create_csr_s;
template<> inline constexpr auto create_spmat_csr<double> = &armpl_spmat_create_csr_d;
template<> inline constexpr auto create_spmat_csr<std::complex<float>> = &armpl_spmat_create_csr_c;
template<> inline constexpr auto create_spmat_csr<std::complex<double>> = &armpl_spmat_create_csr_z;

template<class T>
armpl_status_t (*spmv_exec)(enum armpl_sparse_hint_value, T, armpl_spmat_t, const T *, T, T *);
template<> inline constexpr auto spmv_exec<float> = &armpl_spmv_exec_s;
template<> inline constexpr auto spmv_exec<double> = &armpl_spmv_exec_d;
template<> inline constexpr auto spmv_exec<std::complex<float>> = &armpl_spmv_exec_c;
template<> inline constexpr auto spmv_exec<std::complex<double>> = &armpl_spmv_exec_z;

} // namespace armpl

namespace spblas {

// Vendor specific operation info - could be anything... here I'm just wrapping up Arm PL's sparse matrix type
struct vendor_t {

  armpl_spmat_t pl_mat; // Arm PL matrix type - optimizations under the hood!

  vendor_t(vendor_t&& vt) : pl_mat(vt.pl_mat) {
    vt.pl_mat = nullptr;
  }
  vendor_t() : pl_mat(nullptr) {}

  ~vendor_t() {
    if (pl_mat) {
      std::cout << "destroy vendor_t pl_mat" << std::endl;
      armpl_spmat_destroy(pl_mat);
    }
  }

};

} // namespace spblas
