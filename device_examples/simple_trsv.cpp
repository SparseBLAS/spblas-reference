#include <spblas/spblas.hpp>

#include <fmt/ranges.h>

#include <iostream>

#include "allocator.hpp"

int main(int argc, char** argv) {
  using namespace spblas;

  /**
   *  3  0 0 0
   *  0  1 0 0
   *  0 -3 2 0
   * -5  0 0 4
   */
  int nnz = 6;
  std::vector<float> values{3, 1, -3, 2, -5, 4};
  std::vector<int> rowptr{0, 1, 2, 4, 6};
  std::vector<int> colind{0, 1, 1, 2, 0, 3};
  spblas::index<int> shape(4, 4);

  std::vector<float> rhs{1, 1, 1, 1};

  float* dvalues;
  int *drowptr, *dcolind;
  auto allocator = std::make_shared<const amd_allocator>();

  hipMalloc((void**) &dvalues, sizeof(float) * nnz);
  hipMalloc((void**) &drowptr, sizeof(int) * (shape[0] + 1));
  hipMalloc((void**) &dcolind, sizeof(int) * nnz);
  hipMemcpy(dvalues, values.data(), sizeof(float) * nnz,
             hipMemcpyHostToDevice);
  hipMemcpy(drowptr, rowptr.data(), sizeof(int) * (shape[0] + 1),
             hipMemcpyHostToDevice);
  hipMemcpy(dcolind, colind.data(), sizeof(int) * nnz, hipMemcpyHostToDevice);
  csr_view<float, int> a(dvalues, drowptr, dcolind, shape, nnz);

  float* drhs;
  hipMalloc((void**) &drhs, sizeof(float) * shape[0]);
  hipMemcpy(drhs, rhs.data(), sizeof(float) * shape[0],
             hipMemcpyHostToDevice);
  std::span<float> rhs_span(drhs, shape[0]);

  float* dx;
  hipMalloc((void**) &dx, sizeof(float) * shape[1]);
  std::span<float> x_span(dx, shape[1]);

  triangular_solve_handle_t handle(allocator);
  auto diag_kind = spblas::explicit_diagonal_t();
  auto uplo_kind = spblas::lower_triangle_t();

  triangular_solve_compute(handle, a, uplo_kind, diag_kind, rhs_span, x_span);
  triangular_solve_execute(handle, a, uplo_kind, diag_kind, rhs_span, x_span);

  std::vector<float> x(4);
  hipMemcpy(x.data(), dx, sizeof(float) * shape[1], hipMemcpyDeviceToHost);
  // answer should be the the [1/3 1 2 2/3]'
  for (const auto& v : x) {
    std::cout << v << std::endl;
  }

  hipFree(dvalues);
  hipFree(drowptr);
  hipFree(dcolind);
  hipFree(dx);
  hipFree(drhs);
  return 0;
}
