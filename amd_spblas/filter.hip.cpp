#include <amd_spblas/filter.hpp>
#include <cmath>
#include <cstdint>
#include <hip/hip_runtime.h>
#include <thrust/scan.h>

namespace spblas {
namespace amd {
#define INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(_macro, ...)                 \
  template _macro(float, int32_t, __VA_ARGS__);                                \
  template _macro(double, int32_t, __VA_ARGS__);                               \
  template _macro(float, int64_t, __VA_ARGS__);                                \
  template _macro(double, int64_t, __VA_ARGS__)

// simple impl: one thread per row
template <typename ValueType, typename IndexType, typename Filter>
__global__ void csr_filter_count_row_nnz_kernel(
    std::int64_t n, std::int64_t nnz, const IndexType* __restrict__ row_ptrs,
    const IndexType* __restrict__ col_idxs, const ValueType* __restrict__ vals,
    Filter filter, IndexType* __restrict__ out_row_ptrs) {
  auto tid = static_cast<IndexType>(threadIdx.x + blockDim.x * blockIdx.x);
  if (tid >= n) {
    return;
  }
  IndexType row_nnz = 0;
  for (auto ind = row_ptrs[tid]; ind < row_ptrs[tid + 1]; ind++) {
    row_nnz += filter(tid, col_idxs[ind], vals[ind]);
  }
  out_row_ptrs[tid] = row_nnz;
}

template <typename ValueType, typename IndexType, typename Filter>
__global__ void csr_filter_fill_kernel(
    std::int64_t n, std::int64_t nnz, const IndexType* __restrict__ row_ptrs,
    const IndexType* __restrict__ col_idxs, const ValueType* __restrict__ vals,
    Filter filter, const IndexType* __restrict__ out_row_ptrs,
    IndexType* __restrict__ out_col_idxs, ValueType* __restrict__ out_vals) {
  auto tid = static_cast<IndexType>(threadIdx.x + blockDim.x * blockIdx.x);
  if (tid >= n) {
    return;
  }
  IndexType out_ind = out_row_ptrs[tid];
  for (auto ind = row_ptrs[tid]; ind < row_ptrs[tid + 1]; ind++) {
    if (filter(tid, col_idxs[ind], vals[ind])) {
      out_col_idxs[out_ind] = col_idxs[ind];
      out_vals[out_ind] = vals[ind];
      out_ind++;
    }
  }
}

template <typename ValueType, typename IndexType, typename Filter>
void csr_filter_count_row_nnz(std::int64_t n, std::int64_t nnz,
                              const IndexType* row_ptrs,
                              const IndexType* col_idxs, const ValueType* vals,
                              Filter filter, IndexType* out_row_ptrs,
                              std::int64_t* out_nnz) {
  int block_size = 256;
  int grid_size = (n + block_size - 1) / block_size;
  // some way to dispatch filter (overloading, type check, ...)
  if constexpr (std::is_same_v<Filter, keep_valid_nonzeros>) {
    csr_filter_count_row_nnz_kernel<<<grid_size, block_size>>>(
        n, nnz, row_ptrs, col_idxs, vals,
        [] __device__(auto row, auto col, auto val) {
          return val != decltype(val){0} && !std::isnan(val);
        },
        out_row_ptrs);
  } else if constexpr (std::is_same_v<Filter, get_lower_matrix>) {
    csr_filter_count_row_nnz_kernel<<<grid_size, block_size>>>(
        n, nnz, row_ptrs, col_idxs, vals,
        [include_diag = filter.include_diag] __device__(auto row, auto col,
                                                        auto val) {
          return col < row || (include_diag && col == row);
        },
        out_row_ptrs);
  } else if constexpr (std::is_same_v<Filter, get_upper_matrix>) {
    csr_filter_count_row_nnz_kernel<<<grid_size, block_size>>>(
        n, nnz, row_ptrs, col_idxs, vals,
        [include_diag = filter.include_diag] __device__(auto row, auto col,
                                                        auto val) {
          return row < col || (include_diag && col == row);
        },
        out_row_ptrs);
  }
  thrust::exclusive_scan(thrust::hip::par, out_row_ptrs, out_row_ptrs + n + 1,
                         out_row_ptrs);
  IndexType nnz_result = 0;
  hipMemcpy(&nnz_result, out_row_ptrs + n, sizeof(IndexType),
             hipMemcpyDeviceToHost);
  *out_nnz = static_cast<std::int64_t>(nnz_result);
}

#define CSR_FILTER_COUNT_ROW_NNZ(ValueType, IndexType, Filter)                 \
  void csr_filter_count_row_nnz(                                               \
      std::int64_t n, std::int64_t nnz, const IndexType* row_ptrs,             \
      const IndexType* col_idxs, const ValueType* vals, Filter filter,         \
      IndexType* out_row_ptrs, std::int64_t* out_nnz)

INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(CSR_FILTER_COUNT_ROW_NNZ,
                                          keep_valid_nonzeros);
INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(CSR_FILTER_COUNT_ROW_NNZ,
                                          get_lower_matrix);
INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(CSR_FILTER_COUNT_ROW_NNZ,
                                          get_upper_matrix);

template <typename ValueType, typename IndexType, typename Filter>
void csr_filter_fill(std::int64_t n, std::int64_t nnz,
                     const IndexType* row_ptrs, const IndexType* col_idxs,
                     const ValueType* vals, Filter filter, std::int64_t out_nnz,
                     const IndexType* out_row_ptrs, IndexType* out_col_idxs,
                     ValueType* out_vals) {
  int block_size = 256;
  int grid_size = (n + block_size - 1) / block_size;
  // some way to dispatch filter (overloading, type check, ...)
  if (std::is_same_v<Filter, keep_valid_nonzeros>) {
    csr_filter_fill_kernel<<<grid_size, block_size>>>(
        n, nnz, row_ptrs, col_idxs, vals,
        [] __device__(auto row, auto col, auto val) {
          return val != decltype(val){0} && !std::isnan(val);
        },
        out_row_ptrs, out_col_idxs, out_vals);
  } else if constexpr (std::is_same_v<Filter, get_lower_matrix>) {
    csr_filter_fill_kernel<<<grid_size, block_size>>>(
        n, nnz, row_ptrs, col_idxs, vals,
        [include_diag = filter.include_diag] __device__(auto row, auto col,
                                                        auto val) {
          return col < row || (include_diag && col == row);
        },
        out_row_ptrs, out_col_idxs, out_vals);
  } else if constexpr (std::is_same_v<Filter, get_upper_matrix>) {
    csr_filter_fill_kernel<<<grid_size, block_size>>>(
        n, nnz, row_ptrs, col_idxs, vals,
        [include_diag = filter.include_diag] __device__(auto row, auto col,
                                                        auto val) {
          return row < col || (include_diag && col == row);
        },
        out_row_ptrs, out_col_idxs, out_vals);
  }
}

#define CSR_FILTER_FILL(ValueType, IndexType, Filter)                          \
  void csr_filter_fill(std::int64_t n, std::int64_t nnz,                       \
                       const IndexType* row_ptrs, const IndexType* col_idxs,   \
                       const ValueType* vals, Filter filter,                   \
                       std::int64_t out_nnz, const IndexType* out_row_ptrs,    \
                       IndexType* out_col_idxs, ValueType* out_vals)

INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(CSR_FILTER_FILL, keep_valid_nonzeros);
INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(CSR_FILTER_FILL, get_lower_matrix);
INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(CSR_FILTER_FILL, get_upper_matrix);

} // namespace amd
} // namespace spblas
