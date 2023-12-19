#pragma once

#include <spblas/views/csr_view.hpp>
#include <spblas/vendor/armpl_sparse.hpp>

namespace spblas {

template <typename T, std::integral I = index_t, std::integral O = I>
struct vendor_csr_view {

  T alpha;              // saved scalar
  csr_view<T,I,O> v;    // incorporate the original view for easy access to members
  armpl_spmat_t pl_mat; // this just really gives us a hook into execute functions later on...

  // Ctors create armpl_spmat_t opaque objects with "nocopy" flag - like a view!

  vendor_csr_view(T alpha, csr_view<T, I, O>& v) : alpha(alpha), v(v) {
    auto stat = armpl::create_spmat_csr<T>(&pl_mat, v.shape()[0], v.shape()[1],
                                           (const long *)v.rowptr().data(), (const long *)v.colind().data(),
                                           v.values().data(), ARMPL_SPARSE_CREATE_NOCOPY);
    if (stat != ARMPL_STATUS_SUCCESS) {
      ; // XXX throw error?
    }
  }

  vendor_csr_view(csr_view<T, I, O>& v) : alpha(T(1.0)), v(v) {
    auto stat = armpl::create_spmat_csr<T>(&pl_mat, v.shape()[0], v.shape()[1],
                                           (const long *)v.rowptr().data(), (const long *)v.colind().data(),
                                           v.values().data(), ARMPL_SPARSE_CREATE_NOCOPY);
    if (stat != ARMPL_STATUS_SUCCESS) {
      ;// throw error?
    }
  }

  vendor_csr_view(vendor_csr_view&  vin) = default;

  vendor_csr_view(vendor_csr_view&& vin) : alpha(vin.alpha), v(vin.v), pl_mat(vin.pl_mat) {
    vin.pl_mat = nullptr;
  }

  ~vendor_csr_view() {
    if (pl_mat) {
      std::cout << "destroy view pl_mat" << std::endl;
      armpl_spmat_destroy(pl_mat);
    }
  }

};

} // namespace spblas
