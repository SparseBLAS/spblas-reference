#include <spblas/spblas.hpp>

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

  auto&& [values, rowptr, colind, shape, nnz] =
      generate_csr<float, int>(100, 100, 10);

  float* dvalues;
  int *drowptr, *dcolind;
  MALLOC((void**) &dvalues, sizeof(float) * nnz);
  MALLOC((void**) &drowptr, sizeof(int) * (shape[0] + 1));
  MALLOC((void**) &dcolind, sizeof(int) * nnz);
  COPY(dvalues, values.data(), sizeof(float) * nnz, H2D);
  COPY(drowptr, rowptr.data(), sizeof(int) * (shape[0] + 1), H2D);
  COPY(dcolind, colind.data(), sizeof(int) * nnz, H2D);
  csr_view<float, int> a(dvalues, drowptr, dcolind, shape, nnz);

  // Scale every value of `a` by 5 in place.
  // scale(5.f, a);

  std::vector<float> b(100, 1);
  std::vector<float> c(100, 0);
  float *db, *dc;
  MALLOC((void**) &db, sizeof(float) * 100);
  MALLOC((void**) &dc, sizeof(float) * 100);
  COPY(db, b.data(), sizeof(float) * 100, H2D);
  COPY(dc, c.data(), sizeof(float) * 100, H2D);

  std::span<float> b_span(db, 100);
  std::span<float> c_span(dc, 100);

  float alpha = 2.0f;
  // c = a * alpha * b
  // multiply(a, scaled(alpha, b), c);
  multiply(a, b_span, c_span);

  COPY(c.data(), dc, sizeof(float) * 100, D2H);
  for (int i = 0; i < 100; i++) {
    std::cout << c.at(i) << " ";
  }
  std::cout << std::endl;
  FREE(dvalues);
  FREE(drowptr);
  FREE(dcolind);
  FREE(db);
  FREE(dc);
  return 0;
}
