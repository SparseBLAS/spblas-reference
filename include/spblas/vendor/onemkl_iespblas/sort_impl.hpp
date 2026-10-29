#pragma once

#include <spblas/algorithms/sort_impl.hpp>
#include <spblas/concepts.hpp>
#include <spblas/detail/log.hpp>
#include <spblas/detail/operation_state.hpp>

#include "detail/operand.hpp"
#include "mkl_wrappers.hpp"

//
// oneMKL IE sort_matrix, via mkl_sparse_order
//
//   For a CSR / CSC wrapper (or a matrix_opt of one) with MKL value and index
//   types and values attached, a fresh handle is wrapped around the user's
//   arrays and mkl_sparse_order sorts them in place (for CSC: the CSR arrays
//   of A^T, whose rows are A's columns). Other cases (e.g. a pattern without
//   values) use the reference implementation (algorithms/sort_impl.hpp).
//
//   Workaround for a oneMKL bug (seen in 2026.0, to be fixed in a future
//   release): after mkl_sparse_optimize, a handle created over the user's
//   arrays works on library-allocated copies, and mkl_sparse_order sorts only
//   those, leaving the user's arrays unsorted. So a matrix_opt's cached
//   handle is not used here: the user's arrays are sorted through a fresh
//   handle, and the matrix_opt's optimization data is dropped since its
//   arrays were reordered. Once fixed, the cached handle can be sorted
//   directly and kept.
//
//   mkl_sparse_order is only called here: SpGEMM / SpADD outputs are left in
//   MKL's order. As in the reference, a matrix whose sort_state() is
//   `sorted` is left as it is (no handle is created), and A's sort_state()
//   is `sorted` afterwards.
//

namespace spblas {

namespace __mkl_iespblas {

template <typename A>
concept mkl_sortable =
    __detail::sortable_matrix<A> &&
    sparse_input<decltype(__detail::sortable_base(std::declval<A>()))>;

} // namespace __mkl_iespblas

// More constrained than the reference overload, so it takes precedence.
template <matrix A>
  requires(__detail::sortable_matrix<A> && __mkl_iespblas::mkl_sortable<A>)
void sort_matrix(operation_state& state, A&& a) {
  log_trace("");
  if (a.sort_state() == sort_state::sorted) {
    return;
  }
  auto w = __detail::sortable_base(a);
  using T = tensor_scalar_t<decltype(w)>;
  using I = tensor_index_t<decltype(w)>;

  if (w.size() == 0 || w.values().size() < std::size_t(w.size())) {
    // Nothing to sort, or a pattern only: the reference kernel handles it.
    __detail::sort_slices(state, w);
  } else {
    auto csr = [&] {
      if constexpr (__detail::is_csc_wrapper_v<decltype(w)>) {
        return view::transpose(w);
      } else {
        return w;
      }
    }();
    if (csr.rowptr().size() != std::size_t(csr.shape()[0]) + 1) {
      throw std::invalid_argument(
          "sort_matrix: the matrix's pointer array must have (outer "
          "dimension + 1) entries.");
    }
    __mkl_iespblas::handle_for_t<I> h;
    auto rowptr = csr.rowptr().data();
    __mkl_iespblas::create_csr<T, I>(h, SPARSE_INDEX_BASE_ZERO, csr.shape()[0],
                                     csr.shape()[1], rowptr, rowptr + 1,
                                     csr.colind().data(), csr.values().data());
    __mkl_iespblas::order(h);
  }
  __detail::mark_sorted(a);

  if constexpr (__detail::is_matrix_opt_v<A>) {
    a.state_->invalidate();
  }
}

} // namespace spblas
