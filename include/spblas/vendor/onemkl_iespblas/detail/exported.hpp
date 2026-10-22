#pragma once

#include <stdexcept>
#include <string>

#include <spblas/algorithms/detail/sparse_output.hpp>
#include <spblas/concepts.hpp>

#include "../mkl_wrappers.hpp"

// Copying matrices that MKL created (sp2m / add results, exported with
// export_csr) into the user's CSR / CSC wrappers.

namespace spblas {

namespace __mkl_iespblas {

template <typename T, typename I>
std::size_t exported_nnz(const exported_csr<T, I>& e) {
  std::size_t nnz = 0;
  for (I i = 0; i < e.rows; i++) {
    nnz += e.rows_end[i] - e.rows_start[i];
  }
  return nnz;
}

// Copies an exported matrix into a CSR / CSC wrapper's arrays: the pattern,
// and the values times alpha if `with_values`.
template <typename T, typename I, __detail::compressed_output W>
void copy_exported(const exported_csr<T, I>& e, W&& w, bool with_values,
                   T alpha, const char* operation) {
  using O = tensor_offset_t<W>;
  auto pointers = __detail::outer_pointers(w);
  auto indices = __detail::inner_indices(w);
  auto values = w.values();
  const I base = e.base == SPARSE_INDEX_BASE_ONE ? 1 : 0;
  const std::size_t nnz = exported_nnz(e);

  if (pointers.size() != std::size_t(e.rows) + 1 || indices.size() < nnz ||
      (with_values && values.size() < nnz)) {
    throw std::invalid_argument(
        std::string(operation) +
        ": the output's arrays do not match the result (sized by "
        "get_result_nnz()).");
  }

  O pos = 0;
  pointers[0] = 0;
  for (I i = 0; i < e.rows; i++) {
    for (I p = e.rows_start[i] - base; p < e.rows_end[i] - base; p++) {
      indices[pos] = e.col_indx[p] - base;
      if (with_values) {
        values[pos] = alpha * e.values[p];
      }
      pos++;
    }
    pointers[i + 1] = pos;
  }
  w.update(values, pointers, indices, w.shape(), pos);
}

// Whether a wrapper's pattern is exactly the exported matrix's.
template <typename T, typename I, __detail::compressed_output W>
bool same_pattern(const exported_csr<T, I>& e, W&& w) {
  auto pointers = __detail::outer_pointers(w);
  auto indices = __detail::inner_indices(w);
  const I base = e.base == SPARSE_INDEX_BASE_ONE ? 1 : 0;
  if (pointers.size() != std::size_t(e.rows) + 1 || pointers[0] != 0) {
    return false;
  }
  for (I i = 0; i < e.rows; i++) {
    const I begin = e.rows_start[i] - base;
    const I count = e.rows_end[i] - e.rows_start[i];
    if (pointers[i + 1] - pointers[i] != count) {
      return false;
    }
    for (I p = 0; p < count; p++) {
      if (std::size_t(pointers[i] + p) >= indices.size() ||
          indices[pointers[i] + p] != e.col_indx[begin + p] - base) {
        return false;
      }
    }
  }
  return true;
}

// alpha * the exported values into a wrapper whose pattern is the
// exported one (in the same order).
template <typename T, typename I, __detail::compressed_output W>
void copy_exported_values(const exported_csr<T, I>& e, W&& w, T alpha,
                          const char* operation) {
  auto values = w.values();
  if (values.size() < exported_nnz(e)) {
    throw std::invalid_argument(std::string(operation) +
                                ": the output's values array is not "
                                "allocated (sized by get_result_nnz()).");
  }
  const I base = e.base == SPARSE_INDEX_BASE_ONE ? 1 : 0;
  std::size_t pos = 0;
  for (I i = 0; i < e.rows; i++) {
    for (I p = e.rows_start[i] - base; p < e.rows_end[i] - base; p++) {
      values[pos++] = alpha * e.values[p];
    }
  }
}

} // namespace __mkl_iespblas

} // namespace spblas
