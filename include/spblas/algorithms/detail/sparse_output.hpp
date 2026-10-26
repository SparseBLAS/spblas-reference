#pragma once

#include <memory_resource>
#include <stdexcept>
#include <string>
#include <vector>

#include <spblas/concepts.hpp>
#include <spblas/detail/operation_state.hpp>
#include <spblas/views/inspectors.hpp>

// Helpers for operations with sparse output whose pattern is not known up
// front (SpGEMM, SpADD), used to build the 3-stage API
// (_compute_symbolic / _fill_symbolic / _numeric) on top of a kernel that
// fills pattern and values together.

namespace spblas {

namespace __detail {

template <typename C>
concept compressed_output = is_csr_wrapper_v<C> || is_csc_wrapper_v<C>;

// Pointer array (rowptr / colptr) and index array (colind / rowind) of a CSR
// or CSC wrapper, and the lengths they index.
template <compressed_output C>
auto outer_pointers(C&& c) {
  if constexpr (is_csr_wrapper_v<C>) {
    return c.rowptr();
  } else {
    return c.colptr();
  }
}

template <compressed_output C>
auto inner_indices(C&& c) {
  if constexpr (is_csr_wrapper_v<C>) {
    return c.colind();
  } else {
    return c.rowind();
  }
}

template <compressed_output C>
std::size_t outer_dimension(C&& c) {
  return is_csr_wrapper_v<C> ? c.shape()[0] : c.shape()[1];
}

template <compressed_output C>
std::size_t inner_dimension(C&& c) {
  return is_csr_wrapper_v<C> ? c.shape()[1] : c.shape()[0];
}

// A wrapper of C's format, types and shape over other arrays.
template <compressed_output C, typename V, typename P, typename N>
auto rewrap(C&& c, V&& values, P&& pointers, N&& indices,
            tensor_offset_t<C> nnz) {
  return std::remove_cvref_t<C>(values, pointers, indices, c.shape(), nnz);
}

// _fill_symbolic: run `fill(t)`, a kernel that fills pattern and values,
// on a wrapper `t` sharing C's pattern arrays but with temporary values, so
// only C's pattern is written (C's values may be unallocated).
template <compressed_output C, typename Fill>
void fill_pattern_only(operation_state& state, C&& c, const char* operation,
                       Fill&& fill) {
  using T = tensor_scalar_t<C>;

  if (outer_pointers(c).size() != outer_dimension(c) + 1) {
    throw std::invalid_argument(
        std::string(operation) +
        ": the output's pointer array must have (outer dimension + 1) "
        "entries.");
  }

  std::pmr::vector<T> values(inner_indices(c).size(), state.get_allocator());
  fill(rewrap(c, values, outer_pointers(c), inner_indices(c), c.size()));

  // The kernel set the temporary wrapper's nnz; C's own follows its filled
  // pointer array.
  c.update(c.values(), outer_pointers(c), inner_indices(c), c.shape(),
           outer_pointers(c)[outer_dimension(c)]);
}

// _numeric: run `fill(t)` into temporary arrays for a result with `nnz`
// entries, then scatter the values into C's existing pattern. C's pattern
// may contain entries the result lacks (they become zero), but not the
// reverse.
template <compressed_output C, typename Fill>
void fill_values_only(operation_state& state, C&& c, std::size_t nnz,
                      const char* operation, Fill&& fill) {
  using T = tensor_scalar_t<C>;
  using I = tensor_index_t<C>;
  using O = tensor_offset_t<C>;

  const auto c_pointers = outer_pointers(c);
  const auto c_indices = inner_indices(c);
  const auto c_values = c.values();
  const std::size_t outer = outer_dimension(c);

  // The stored entries are those C's pointer array covers; the index and
  // value arrays may be longer.
  if (c_pointers.size() != outer + 1 ||
      c_indices.size() < std::size_t(c_pointers[outer]) ||
      c_values.size() < std::size_t(c_pointers[outer])) {
    throw std::invalid_argument(
        std::string(operation) +
        ": the output's pattern must be filled (e.g. by _fill_symbolic) "
        "and its values array allocated.");
  }

  auto allocator = state.get_allocator();
  std::pmr::vector<T> t_values(nnz, allocator);
  std::pmr::vector<O> t_pointers(outer + 1, allocator);
  std::pmr::vector<I> t_indices(nnz, allocator);
  fill(rewrap(c, t_values, t_pointers, t_indices, O(nnz)));

  // Per outer index: stage the result's entries densely, read them out
  // through C's pattern, then check every result entry was consumed.
  enum : char { empty = 0, staged = 1, consumed = 2 };
  std::pmr::vector<T> dense(inner_dimension(c), allocator);
  std::pmr::vector<char> mark(inner_dimension(c), empty, allocator);

  for (std::size_t o = 0; o < outer; o++) {
    for (auto p = t_pointers[o]; p < t_pointers[o + 1]; p++) {
      dense[t_indices[p]] = t_values[p];
      mark[t_indices[p]] = staged;
    }
    for (auto q = c_pointers[o]; q < c_pointers[o + 1]; q++) {
      auto j = c_indices[q];
      c_values[q] = mark[j] == empty ? T(0) : dense[j];
      if (mark[j] == staged) {
        mark[j] = consumed;
      }
    }
    bool contained = true;
    for (auto p = t_pointers[o]; p < t_pointers[o + 1]; p++) {
      contained = contained && mark[t_indices[p]] == consumed;
      mark[t_indices[p]] = empty;
    }
    if (!contained) {
      throw std::invalid_argument(
          std::string(operation) +
          ": the output's pattern does not contain the result's pattern; "
          "fill it with _fill_symbolic first.");
    }
  }
}

} // namespace __detail

} // namespace spblas
