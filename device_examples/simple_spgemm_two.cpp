#include <iostream>

#include <spblas/spblas.hpp>

#include <fmt/core.h>
#include <fmt/ranges.h>

class cuda_allocator : public spblas::allocator {
public:
  void alloc(void** ptrptr, size_t size) const override {
    cudaMalloc(ptrptr, size);
  }

  void free(void* ptr) const override {
    cudaFree(ptr);
  }
};

int main(int argc, char** argv) {
  using namespace spblas;
  namespace md = spblas::__mdspan;

  spblas::index_t m = 100;
  spblas::index_t n = 10;
  spblas::index_t k = 100;
  spblas::index_t nnz = 100;

  auto allocator = std::make_shared<cuda_allocator>();
  spblas::spgemm_handle_t spgemm_handle(allocator);

  auto&& [a_values, a_rowptr, a_colind, a_shape, as] =
      generate_csr<float, int>(m, k, nnz);
  auto&& [b_values, b_rowptr, b_colind, b_shape, bs] =
      generate_csr<float, int>(k, n, nnz);

  float *da_values, *db_values;
  int *da_rowptr, *da_colind, *db_rowptr, *db_colind;
  cudaMalloc((void**) &da_values, sizeof(float) * nnz);
  cudaMemcpy((void**) da_values, a_values.data(), sizeof(float) * nnz,
             cudaMemcpyHostToDevice);
  std::span<float> a_values_span(da_values, nnz);
  cudaMalloc((void**) &da_rowptr, sizeof(int) * (m + 1));
  cudaMemcpy(da_rowptr, a_rowptr.data(), sizeof(int) * (m + 1),
             cudaMemcpyHostToDevice);
  std::span<int> a_rowptr_span(da_rowptr, nnz);
  cudaMalloc((void**) &da_colind, sizeof(int) * nnz);
  cudaMemcpy(da_colind, a_colind.data(), sizeof(int) * nnz,
             cudaMemcpyHostToDevice);
  std::span<int> a_colind_span(da_colind, nnz);
  csr_view<float, int> a(a_values_span, a_rowptr_span, a_colind_span, a_shape,
                         nnz);

  cudaMalloc((void**) &db_values, sizeof(float) * nnz);
  cudaMemcpy(db_values, b_values.data(), sizeof(float) * nnz,
             cudaMemcpyHostToDevice);
  std::span<float> b_values_span(db_values, nnz);
  cudaMalloc((void**) &db_rowptr, sizeof(int) * (k + 1));
  cudaMemcpy(db_rowptr, b_rowptr.data(), sizeof(int) * (k + 1),
             cudaMemcpyHostToDevice);
  std::span<int> b_rowptr_span(db_rowptr, nnz);
  cudaMalloc((void**) &db_colind, sizeof(int) * nnz);
  cudaMemcpy(db_colind, b_colind.data(), sizeof(int) * nnz,
             cudaMemcpyHostToDevice);
  std::span<int> b_colind_span(db_colind, nnz);
  csr_view<float, int> b(b_values_span, b_rowptr_span, b_colind_span, b_shape,
                         nnz);

  int* dc_rowptr;
  cudaMalloc((void**) &dc_rowptr, sizeof(int) * (m + 1));
  // std::span<int> c_rowptr_span(dc_rowptr, m+1);

  csr_view<float, int> c(nullptr, dc_rowptr, nullptr, {m, n}, 0);
  multiply_inspect(spgemm_handle, a, b, c);
  multiply_compute(spgemm_handle, a, b, c);

  float* dc_values;
  int* dc_colind;
  cudaMalloc((void**) &dc_values, spgemm_handle.result_nnz() * sizeof(float));
  cudaMalloc((void**) &dc_colind, spgemm_handle.result_nnz() * sizeof(int));

  std::span<int> c_rowptr_span(dc_rowptr, m + 1);
  std::span<int> c_colind_span(dc_colind, spgemm_handle.result_nnz());
  std::span<float> c_values_span(dc_values, spgemm_handle.result_nnz());
  c.update(c_values_span, c_rowptr_span, c_colind_span, {m, n},
           (int) spgemm_handle.result_nnz());

  multiply_execute(spgemm_handle, a, b, c);

  std::vector<int> c_rowptr(m + 1);
  cudaMemcpy(c_rowptr.data(), dc_rowptr, sizeof(int) * (m + 1),
             cudaMemcpyDeviceToHost);
  for (int i = 0; i < m + 1; i++) {
    std::cout << c_rowptr.at(i) << " ";
  }
  std::cout << std::endl;
  cudaFree(da_values);
  cudaFree(da_rowptr);
  cudaFree(da_colind);
  cudaFree(db_values);
  cudaFree(db_rowptr);
  cudaFree(db_colind);
  cudaFree(dc_rowptr);
  cudaFree(dc_values);
  cudaFree(dc_colind);

  return 0;
}
