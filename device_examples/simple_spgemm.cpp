#include <spblas/spblas.hpp>

#include <fmt/core.h>
#include <fmt/ranges.h>
#include <iostream>

// TODO: create an allocator? also use custom data type to handle lifetime?
#ifdef SPBLAS_ENABLE_HIPSPARSE
#define FREE hipFree
#define MALLOC hipMalloc
#define COPY hipMemcpy
#define D2H hipMemcpyDeviceToHost
#define H2D hipMemcpyHostToDevice
#else
#define FREE cudaFree
#define MALLOC cudaMalloc
#define COPY cudaMemcpy
#define D2H cudaMemcpyDeviceToHost
#define H2D cudaMemcpyHostToDevice
#endif

int main(int argc, char** argv) {
  using namespace spblas;
  namespace md = spblas::__mdspan;

  spblas::index_t m = 100;
  spblas::index_t n = 10;
  spblas::index_t k = 100;
  spblas::index_t nnz = 100;

  auto&& [a_values, a_rowptr, a_colind, a_shape, as] =
      generate_csr<float, int>(m, k, nnz);
  auto&& [b_values, b_rowptr, b_colind, b_shape, bs] =
      generate_csr<float, int>(k, n, nnz);

  float *da_values, *db_values;
  int *da_rowptr, *da_colind, *db_rowptr, *db_colind;
  MALLOC((void**) &da_values, sizeof(float) * nnz);
  COPY(da_values, a_values.data(), sizeof(float) * nnz, H2D);
  std::span<float> a_values_span(da_values, nnz);
  MALLOC((void**) &da_rowptr, sizeof(int) * (m + 1));
  COPY(da_rowptr, a_rowptr.data(), sizeof(int) * (m + 1), H2D);
  std::span<int> a_rowptr_span(da_rowptr, nnz);
  MALLOC((void**) &da_colind, sizeof(int) * nnz);
  COPY(da_colind, a_colind.data(), sizeof(int) * nnz, H2D);
  std::span<int> a_colind_span(da_colind, nnz);
  csr_view<float, int> a(a_values_span, a_rowptr_span, a_colind_span, a_shape,
                         nnz);

  MALLOC((void**) &db_values, sizeof(float) * nnz);
  COPY(db_values, b_values.data(), sizeof(float) * nnz, H2D);
  std::span<float> b_values_span(db_values, nnz);
  MALLOC((void**) &db_rowptr, sizeof(int) * (k + 1));
  COPY(db_rowptr, b_rowptr.data(), sizeof(int) * (k + 1), H2D);
  std::span<int> b_rowptr_span(db_rowptr, nnz);
  MALLOC((void**) &db_colind, sizeof(int) * nnz);
  COPY(db_colind, b_colind.data(), sizeof(int) * nnz, H2D);
  std::span<int> b_colind_span(db_colind, nnz);
  csr_view<float, int> b(b_values_span, b_rowptr_span, b_colind_span, b_shape,
                         nnz);

  int* dc_rowptr;
  MALLOC((void**) &dc_rowptr, sizeof(int) * (m + 1));
  std::span<int> c_rowptr_span(dc_rowptr, m + 1);

  csr_view<float, int> c(nullptr, dc_rowptr, nullptr, {m, n}, 0);
  multiply(a, b, c);

  std::vector<int> c_rowptr(m + 1);
  COPY(c_rowptr.data(), dc_rowptr, sizeof(int) * (m + 1), D2H);

  for (int i = 0; i < m + 1; i++) {
    std::cout << c_rowptr.at(i) << " ";
  }
  std::cout << std::endl;

  FREE(da_values);
  FREE(da_rowptr);
  FREE(da_colind);
  FREE(db_values);
  FREE(db_rowptr);
  FREE(db_colind);
  FREE(dc_rowptr);
  FREE(c.values().data());
  FREE(c.colind().data());
  return 0;
}
