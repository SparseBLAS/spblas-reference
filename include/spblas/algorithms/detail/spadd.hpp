#pragma once

#include <memory_resource>

#include <stdexcept>

#include <spblas/algorithms/detail/checked_nnz.hpp>
#include <spblas/backend/backend.hpp>
#include <spblas/backend/csr_builder.hpp>
#include <spblas/backend/spa_accumulator.hpp>
#include <spblas/concepts.hpp>
#include <spblas/views/inspectors.hpp>

// Reference SpADD kernels, C = A + B, in spblas::__detail:
//   spadd_nnz(a, b, c, allocator)   number of entries in C's pattern (union
//                                   of A's and B's patterns)
//   spadd_fill(a, b, c, allocator)  fills C's arrays (sized by spadd_nnz)
//
//   Their scratch storage comes from `allocator`, a
//   std::pmr::polymorphic_allocator<>; operations pass their
//   operation_state's.
//
//   A, B, C all CSR (A, B row-lookupable), or all CSC (A, B
//   column-lookupable); matrix_opt of either works too. Mixed orientations
//   are not supported.

namespace spblas {

namespace __detail {

template <typename A, typename B, typename C>
concept spadd_by_rows = __backend::row_lookupable<A> &&
                        __backend::row_lookupable<B> && is_csr_wrapper_v<C>;

template <typename A, typename B, typename C>
concept spadd_by_columns =
    __backend::column_lookupable<A> && __backend::column_lookupable<B> &&
    is_csc_wrapper_v<C>;

// Row i of a CSR-style input, or column i of a CSC-style input.
template <bool ByColumns, typename M, typename Index>
auto lookup_outer(M&& m, Index i) {
  if constexpr (ByColumns) {
    return __backend::lookup_column(m, i);
  } else {
    return __backend::lookup_row(m, i);
  }
}

template <matrix A, matrix B, matrix C>
void check_spadd_shapes(A&& a, B&& b, C&& c) {
  if (__backend::shape(a) != __backend::shape(b) ||
      __backend::shape(b) != __backend::shape(c)) {
    throw std::invalid_argument("add: matrix dimensions are incompatible.");
  }
}

template <matrix A, matrix B, matrix C>
  requires(spadd_by_rows<A, B, C> || spadd_by_columns<A, B, C>)
tensor_offset_t<C> spadd_nnz(A&& a, B&& b, C&& c,
                             std::pmr::polymorphic_allocator<> allocator) {
  check_spadd_shapes(a, b, c);

  using I = tensor_index_t<C>;
  constexpr bool by_columns = is_csc_wrapper_v<C>;
  const I outer = __backend::shape(c)[by_columns ? 1 : 0];
  const I inner = __backend::shape(c)[by_columns ? 0 : 1];

  std::size_t nnz = 0;
  __backend::spa_set<I> c_slice(inner, allocator);

  for (I o = 0; o < outer; o++) {
    c_slice.clear();
    for (auto&& [j, _] : lookup_outer<by_columns>(a, o)) {
      c_slice.insert(j);
    }
    for (auto&& [j, _] : lookup_outer<by_columns>(b, o)) {
      c_slice.insert(j);
    }
    nnz += c_slice.size();
  }
  return checked_nnz<tensor_offset_t<C>>(nnz);
}

template <matrix A, matrix B, matrix C>
  requires(spadd_by_rows<A, B, C> || spadd_by_columns<A, B, C>)
void spadd_fill(A&& a, B&& b, C&& c,
                std::pmr::polymorphic_allocator<> allocator) {
  check_spadd_shapes(a, b, c);

  using T = tensor_scalar_t<C>;
  using I = tensor_index_t<C>;
  constexpr bool by_columns = is_csc_wrapper_v<C>;
  const I outer = __backend::shape(c)[by_columns ? 1 : 0];
  const I inner = __backend::shape(c)[by_columns ? 0 : 1];

  __backend::spa_accumulator<T, I> c_slice(inner, allocator);
  auto c_builder = [&] {
    if constexpr (by_columns) {
      return __backend::csc_builder(c);
    } else {
      return __backend::csr_builder(c);
    }
  }();

  for (I o = 0; o < outer; o++) {
    c_slice.clear();
    for (auto&& [j, v] : lookup_outer<by_columns>(a, o)) {
      c_slice[j] += v;
    }
    for (auto&& [j, v] : lookup_outer<by_columns>(b, o)) {
      c_slice[j] += v;
    }
    c_slice.sort();

    try {
      if constexpr (by_columns) {
        c_builder.insert_column(o, c_slice.get());
      } else {
        c_builder.insert_row(o, c_slice.get());
      }
    } catch (...) {
      throw std::runtime_error(
          "add: ran out of memory. The output's arrays are too small.");
    }
  }

  if constexpr (by_columns) {
    c.update(c.values(), c.colptr(), c.rowind(), c.shape(), c.colptr()[outer]);
  } else {
    c.update(c.values(), c.rowptr(), c.colind(), c.shape(), c.rowptr()[outer]);
  }
}

} // namespace __detail

} // namespace spblas
