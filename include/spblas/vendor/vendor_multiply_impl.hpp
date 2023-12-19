#pragma once

#include <spblas/concepts.hpp>
#include <spblas/detail/bind_info.hpp>
#include <spblas/vendor/armpl_sparse.hpp>
#include <iostream>

namespace spblas {

// multiply just using the armpl "view" object - no extra memory involved
template<typename T>
void multiply(vendor_csr_view<T, size_t, size_t>&& a, std::vector<T>& b, std::vector<T>& c) {
  std::cout << __FILE__ << ":" << __func__ << ":" << __LINE__ << std::endl;
  T beta = 0.0;
  auto stat = armpl::spmv_exec<T>(ARMPL_SPARSE_OPERATION_NOTRANS, a.alpha, a.pl_mat,
                                  b.data(), beta, c.data());
  if (stat != ARMPL_STATUS_SUCCESS) {
    ; // throw error? XXX
  }
}


// output from `scaled` is effectively armpl_spmat_t with "nocopy"...
// `_inspect` creates a "allow copy" version as an info object to be used preferentially in the multiply later on
template <typename T>
operation_info_t multiply_inspect(vendor_csr_view<T, size_t, size_t>&& a, std::vector<T>& b, std::vector<T>& c) {
  operation_info_t op_info = multiply_inspect(a.v, b, c); // first perform a "basic" inspect on the view?
  std::shared_ptr<vendor_t> pl_info = std::make_shared<vendor_t>();
  auto stat = armpl::create_spmat_csr<T>(&pl_info->pl_mat, a.v.shape()[0], a.v.shape()[1],
                                         (const armpl_int_t *)a.v.rowptr().data(), (const armpl_int_t *)a.v.colind().data(),
                                         a.v.values().data(), 0); // allow ourselves to create a copy here
  if (stat != ARMPL_STATUS_SUCCESS) {
    ;// throw error?
  }

  // optimize
  armpl_spmat_hint(pl_info->pl_mat, ARMPL_SPARSE_HINT_SPMV_INVOCATIONS, ARMPL_SPARSE_INVOCATIONS_MANY);
  armpl_spmv_optimize(pl_info->pl_mat);

  op_info.add_vendor_info(pl_info);
  return op_info;
}


// multiply using the bound object - unpack the optimized pl_mat to get best performance
template <typename T>
void multiply(bind_info<vendor_csr_view<T>>&& bv, std::vector<T>& b, std::vector<T>& c) {
  std::cout << __FILE__ << ":" << __func__ << ":" << __LINE__ << std::endl;
  T beta = 0.0;
  T alpha = bv.a.alpha;
  auto pl_mat = bv.info.get_vendor_info()->pl_mat;
  auto stat = armpl::spmv_exec<T>(ARMPL_SPARSE_OPERATION_NOTRANS, alpha, pl_mat, b.data(), beta, c.data());
  if (stat != ARMPL_STATUS_SUCCESS) {
    ; // throw error?
  }
}

} // namespace spblas
