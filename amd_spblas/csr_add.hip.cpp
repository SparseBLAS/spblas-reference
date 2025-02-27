#include <amd_spblas/csr_add.hpp>
#include <cstdint>
#include <hip/hip_runtime.h>
#include <thrust/scan.h>

namespace spblas {
namespace amd {
#define INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(_macro)                      \
  template _macro(float, int32_t);                                             \
  template _macro(double, int32_t);                                            \
  template _macro(float, int64_t);                                             \
  template _macro(double, int64_t)

// simple impl: one thread per row
template <typename IndexType>
__global__ void
csr_add_count_row_nnz_kernel(std::int64_t n, std::int64_t a_nnz,
                             const IndexType* __restrict__ a_row_ptrs,
                             const IndexType* __restrict__ a_col_idxs,
                             std::int64_t b_nnz,
                             const IndexType* __restrict__ b_row_ptrs,
                             const IndexType* __restrict__ b_col_idxs,
                             IndexType* __restrict__ out_row_ptrs) {
  auto tid = threadIdx.x + blockDim.x * blockIdx.x;
  if (tid >= n) {
    return;
  }
  auto a_ind = a_row_ptrs[tid];
  auto b_ind = b_row_ptrs[tid];
  auto a_end = a_row_ptrs[tid + 1];
  auto b_end = b_row_ptrs[tid + 1];
  IndexType nnz = 0;
  while (a_ind < a_end && b_ind < b_end) {
    auto a_col = a_col_idxs[a_ind];
    auto b_col = b_col_idxs[b_ind];
    a_ind += (a_col <= b_col);
    b_ind += (b_col <= a_col);
    nnz++;
  }
  nnz += (b_end - b_ind) + (a_end - a_ind);
  out_row_ptrs[tid] = nnz;
}

template <typename ValueType, typename IndexType>
__global__ void
csr_add_fill_kernel(std::int64_t n, std::int64_t a_nnz,
                    const IndexType* __restrict__ a_row_ptrs,
                    const IndexType* __restrict__ a_col_idxs,
                    const ValueType* __restrict__ a_vals, std::int64_t b_nnz,
                    const IndexType* __restrict__ b_row_ptrs,
                    const IndexType* __restrict__ b_col_idxs,
                    const ValueType* __restrict__ b_vals, std::int64_t out_nnz,
                    const IndexType* __restrict__ out_row_ptrs,
                    IndexType* __restrict__ out_col_idxs,
                    ValueType* __restrict__ out_vals) {
  auto tid = threadIdx.x + blockDim.x * blockIdx.x;
  if (tid >= n) {
    return;
  }
  auto a_ind = a_row_ptrs[tid];
  auto b_ind = b_row_ptrs[tid];
  auto a_end = a_row_ptrs[tid + 1];
  auto b_end = b_row_ptrs[tid + 1];
  IndexType ind = out_row_ptrs[tid];
  while (a_ind < a_end && b_ind < b_end) {
    auto a_col = a_col_idxs[a_ind];
    auto b_col = b_col_idxs[b_ind];
    if (a_col < b_col) {
      out_col_idxs[ind] = a_col;
      out_vals[ind] = a_vals[a_ind];
      a_ind++;
    } else if (a_col > b_col) {
      out_col_idxs[ind] = b_col;
      out_vals[ind] = b_vals[b_ind];
      b_ind++;
    } else {
      out_col_idxs[ind] = a_col;
      out_vals[ind] = a_vals[a_ind] + b_vals[b_ind];
      a_ind++;
      b_ind++;
    }
    ind++;
  }
  // only one of following is executed
  for (; a_ind < a_end; a_ind++, ind++) {
    out_vals[ind] = a_vals[ind];
    out_col_idxs[ind] = a_col_idxs[ind];
  }
  for (; b_ind < b_end; b_ind++, ind++) {
    out_vals[ind] = b_vals[ind];
    out_col_idxs[ind] = b_col_idxs[ind];
  }
}

/**
 * Count the nnz per row of two matrices addition.
 */
template <typename IndexType>
void csr_add_count_row_nnz(std::int64_t n, std::int64_t a_nnz,
                           const IndexType* a_row_ptrs,
                           const IndexType* a_col_idxs, std::int64_t b_nnz,
                           const IndexType* b_row_ptrs,
                           const IndexType* b_col_idxs, IndexType* out_row_ptrs,
                           std::int64_t* out_nnz) {
  int block_size = 256;
  int grid_size = (n + block_size - 1) / block_size;
  csr_add_count_row_nnz_kernel<<<grid_size, block_size>>>(
      n, a_nnz, a_row_ptrs, a_col_idxs, b_nnz, b_row_ptrs, b_col_idxs,
      out_row_ptrs);
  thrust::exclusive_scan(thrust::hip::par, out_row_ptrs, out_row_ptrs + n + 1,
                         out_row_ptrs);
  IndexType nnz = 0;
  hipMemcpy(&nnz, out_row_ptrs + n, sizeof(IndexType), hipMemcpyDeviceToHost);
  *out_nnz = static_cast<std::int64_t>(nnz);
}

#define CSR_ADD_COUNT_ROW_NNZ(index_type_)                                     \
  void csr_add_count_row_nnz(                                                  \
      std::int64_t n, std::int64_t a_nnz, const index_type_* a_row_ptrs,       \
      const index_type_* a_col_idxs, std::int64_t b_nnz,                       \
      const index_type_* b_row_ptrs, const index_type_* b_col_idxs,            \
      index_type_* out_row_ptrs, std::int64_t* out_nnz)

template CSR_ADD_COUNT_ROW_NNZ(std::int32_t);
template CSR_ADD_COUNT_ROW_NNZ(std::int64_t);

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
                  ValueType* out_vals) {
  int block_size = 256;
  int grid_size = (n + block_size - 1) / block_size;
  csr_add_fill_kernel<<<grid_size, block_size>>>(
      n, a_nnz, a_row_ptrs, a_col_idxs, a_vals, b_nnz, b_row_ptrs, b_col_idxs,
      b_vals, out_nnz, out_row_ptrs, out_col_idxs, out_vals);
}

#define CSR_ADD_FILL(value_type_, index_type_)                                 \
  void csr_add_fill(std::int64_t n, std::int64_t a_nnz,                        \
                    const index_type_* a_row_ptrs,                             \
                    const index_type_* a_col_idxs, const value_type_* a_vals,  \
                    std::int64_t b_nnz, const index_type_* b_row_ptrs,         \
                    const index_type_* b_col_idxs, const value_type_* b_vals,  \
                    std::int64_t out_nnz, const index_type_* out_row_ptrs,     \
                    index_type_* out_col_idxs, value_type_* out_vals)

INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(CSR_ADD_FILL);

} // namespace amd
} // namespace spblas
