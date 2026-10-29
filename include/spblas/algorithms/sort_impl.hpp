#pragma once

#include <algorithm>
#include <iterator>
#include <memory_resource>
#include <numeric>
#include <stdexcept>
#include <utility>

#include <spblas/algorithms/detail/sparse_output.hpp>
#include <spblas/concepts.hpp>
#include <spblas/detail/api_concepts.hpp>
#include <spblas/detail/log.hpp>
#include <spblas/detail/operation_state.hpp>
#include <spblas/views/inspectors.hpp>

//
// Reference sort_matrix (SparseBLAS paper: a function that modifies the
// user's data by sorting it)
//
//   sort_matrix([policy,] state, A)
//     Sorts A's indices within every row (CSR) or column (CSC) in place,
//     moving the values with them. The matrix itself does not change. This
//     is a batched sort by key over the rows (columns): keys are the indices,
//     the values are passed along (__detail::batched_sort_by_key).
//
//   A: a CSR or CSC wrapper, or a matrix_opt of one (whose optimization data
//   is then dropped, as the arrays it was built from were reordered). If A's
//   values array is empty (pattern only), only the indices are sorted.
//   The stored entries are those the pointer array covers; the index and
//   value arrays may be longer. The index and value types must not be
//   const (a csr_wrapper<const T> does not match).
//
//   A matrix whose sort_state() is `sorted` is left as it is; afterwards A's
//   sort_state() is `sorted` (for a matrix_opt: in the wrapper its copies
//   share).
//
//   No operation guarantees sorted output: the order within rows (CSR) or
//   columns (CSC) of SpGEMM / SpADD outputs is backend-defined (the output's
//   sort_state() says what the backend knows), and sort_matrix gives sorted
//   output where it is needed.
//

namespace spblas {

namespace __detail {

// A CSR / CSC wrapper whose index and value arrays can be written.
template <typename W>
concept writable_compressed =
    compressed_output<W> && !std::is_const_v<tensor_scalar_t<W>> &&
    !std::is_const_v<tensor_index_t<W>>;

template <typename M>
constexpr bool is_sortable_matrix() {
  using U = std::remove_cvref_t<M>;
  if constexpr (is_matrix_opt_v<U>) {
    return writable_compressed<decltype(std::declval<U>().base())>;
  } else {
    return writable_compressed<U>;
  }
}

template <typename M>
concept sortable_matrix = is_sortable_matrix<M>();

// The wrapper itself, or a matrix_opt's.
template <sortable_matrix M>
auto sortable_base(M&& m) {
  if constexpr (is_matrix_opt_v<M>) {
    return m.base();
  } else {
    return m;
  }
}

// Batched (segmented) sort by key: for every segment s, sorts
// keys[offsets[s], offsets[s + 1]) into increasing order and applies the same
// permutation to values (if `values` is non-empty). Already sorted segments
// are left as they are.
//
// A segment's permuted keys and values are both gathered into scratch before
// either is written back, so if an allocation throws, every segment is either
// sorted or untouched: the matrix the arrays represent never changes.
template <typename Offsets, typename Keys, typename Values>
void batched_sort_by_key(Offsets&& offsets, Keys&& keys, Values&& values,
                         std::pmr::polymorphic_allocator<> allocator) {
  using K = std::remove_cvref_t<decltype(keys[0])>;
  using V = std::remove_cvref_t<decltype(values[0])>;
  const bool with_values = std::size(values) != 0;

  std::pmr::vector<std::size_t> order(allocator);
  std::pmr::vector<K> sorted_keys(allocator);
  std::pmr::vector<V> sorted_values(allocator);

  for (std::size_t s = 0; s + 1 < std::size(offsets); s++) {
    const std::size_t first = offsets[s];
    const std::size_t last = offsets[s + 1];
    if (std::is_sorted(std::begin(keys) + first, std::begin(keys) + last)) {
      continue;
    }
    order.resize(last - first);
    std::iota(order.begin(), order.end(), first);
    std::sort(order.begin(), order.end(),
              [&](std::size_t l, std::size_t r) { return keys[l] < keys[r]; });

    // All allocations happen here, before the user's arrays are written.
    sorted_keys.resize(order.size());
    if (with_values) {
      sorted_values.resize(order.size());
    }
    for (std::size_t q = 0; q < order.size(); q++) {
      sorted_keys[q] = keys[order[q]];
      if (with_values) {
        sorted_values[q] = values[order[q]];
      }
    }

    std::copy(sorted_keys.begin(), sorted_keys.end(), std::begin(keys) + first);
    if (with_values) {
      std::copy(sorted_values.begin(), sorted_values.end(),
                std::begin(values) + first);
    }
  }
}

// A CSR / CSC matrix as a batched sort by key: segments are rows (CSR) or
// columns (CSC), keys the indices, values the values.
template <compressed_output W>
void sort_slices(operation_state& state, W&& w) {
  auto pointers = outer_pointers(w);
  auto indices = inner_indices(w);
  auto values = w.values();
  if (pointers.size() != outer_dimension(w) + 1) {
    throw std::invalid_argument(
        "sort_matrix: the matrix's pointer array must have (outer dimension "
        "+ 1) entries.");
  }
  // Only the stored entries are sorted; the arrays may be longer. An empty
  // values array means a pattern only, whose indices alone are sorted.
  const std::size_t stored = pointers[pointers.size() - 1];
  if (indices.size() < stored ||
      (values.size() != 0 && values.size() < stored)) {
    throw std::invalid_argument(
        "sort_matrix: the index and value arrays must cover every stored "
        "entry (or the value array be empty).");
  }
  batched_sort_by_key(pointers, indices, values, state.get_allocator().host());
}

// Records that `m` (a wrapper or a matrix_opt of one) is sorted; a const
// wrapper cannot record it.
template <sortable_matrix M>
void mark_sorted(M&& m) {
  if constexpr (is_matrix_opt_v<M> ||
                !std::is_const_v<std::remove_reference_t<M>>) {
    m.set_sort_state(spblas::sort_state::sorted);
  }
}

} // namespace __detail

template <matrix A>
  requires(__detail::sortable_matrix<A>)
void sort_matrix(operation_state& state, A&& a) {
  log_trace("");
  if (a.sort_state() == sort_state::sorted) {
    return;
  }
  __detail::sort_slices(state, __detail::sortable_base(a));
  __detail::mark_sorted(a);
  if constexpr (__detail::is_matrix_opt_v<A>) {
    a.state_->invalidate();
  }
}

// Optional leading execution policy. The reference implementation runs
// sequentially and ignores it.
template <__detail::execution_policy P, typename... Args>
void sort_matrix(P&& policy, operation_state& state, Args&&... args) {
  log_trace("");
  sort_matrix(state, std::forward<Args>(args)...);
}

} // namespace spblas
