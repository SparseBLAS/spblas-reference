#pragma once

#include <cstdint>

namespace spblas {
struct keep_valid_nonzeros {};
struct get_lower_matrix {
  bool include_diag;
};
struct get_upper_matrix {
  bool include_diag;
};

namespace amd {
/**
 * Count the nnz per row after filter.
 */
template <typename ValueType, typename IndexType, typename Filter>
void csr_filter_count_row_nnz(std::int64_t n, std::int64_t nnz,
                              const IndexType* row_ptrs,
                              const IndexType* col_idxs, const ValueType* vals,
                              Filter filter, IndexType* out_row_ptrs,
                              std::int64_t* out_nnz);
/**
 * fill the column and values after filter
 */
template <typename ValueType, typename IndexType, typename Filter>
void csr_filter_fill(std::int64_t n, std::int64_t nnz,
                     const IndexType* row_ptrs, const IndexType* col_idxs,
                     const ValueType* vals, Filter filter, std::int64_t out_nnz,
                     const IndexType* out_row_ptrs, IndexType* out_col_idxs,
                     ValueType* out_vals);

} // namespace amd
} // namespace spblas
