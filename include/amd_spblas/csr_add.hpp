#pragma once

#include <cstdint>

namespace spblas {
namespace amd {

/**
 * Count the nnz per row of two matrices addition.
 */
template <typename IndexType>
void csr_add_count_row_nnz(std::int64_t n, std::int64_t a_nnz,
                           const IndexType* a_row_ptrs,
                           const IndexType* a_col_idxs, std::int64_t b_nnz,
                           const IndexType* b_row_ptrs,
                           const IndexType* b_col_idxs, IndexType* out_row_ptrs,
                           std::int64_t* out_nnz);
/**
 * fill the column and values from two matrices addition.
 */
template <typename ValueType, typename IndexType>
void csr_add_fill(std::int64_t n, std::int64_t a_nnz,
                  const IndexType* a_row_ptrs, const IndexType* a_col_idxs,
                  const ValueType* a_vals, std::int64_t b_nnz,
                  const IndexType* b_row_ptrs, const IndexType* b_col_idxs,
                  const ValueType* b_vals, std::int64_t out_nnz,
                  const IndexType* out_row_ptrs, IndexType* out_col_idxs,
                  ValueType* out_vals);

} // namespace amd
} // namespace spblas
