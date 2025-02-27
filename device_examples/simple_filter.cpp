#include <spblas/spblas.hpp>

#include <fmt/ranges.h>

#include <iostream>

#include "allocator.hpp"

int main(int argc, char** argv) {
  using namespace spblas;

  /**
   *  3  0 0 0
   *  nan  1 0 0
   *  {0} -3 2 0
   * -5  0 0 4
   */
  int a_nnz = 8;
  std::vector<float> a_values{3, std::nanf(""), 1, 0, -3, 2, -5, 4};
  std::vector<int> a_rowptr{0, 1, 3, 6, 8};
  std::vector<int> a_colind{0, 0, 1, 0, 1, 2, 0, 3};
  spblas::index<int> shape(4, 4);

  float* da_values;
  int *da_rowptr, *da_colind;
  auto allocator = std::make_shared<const amd_allocator>();

  hipMalloc((void**) &da_values, sizeof(float) * a_nnz);
  hipMalloc((void**) &da_rowptr, sizeof(int) * (shape[0] + 1));
  hipMalloc((void**) &da_colind, sizeof(int) * a_nnz);
  hipMemcpy(da_values, a_values.data(), sizeof(float) * a_nnz,
             hipMemcpyHostToDevice);
  hipMemcpy(da_rowptr, a_rowptr.data(), sizeof(int) * (shape[0] + 1),
             hipMemcpyHostToDevice);
  hipMemcpy(da_colind, a_colind.data(), sizeof(int) * a_nnz,
             hipMemcpyHostToDevice);
  csr_view<float, int> a(da_values, da_rowptr, da_colind, shape, a_nnz);

  float* db_values;
  int *db_rowptr, *db_colind;

  hipMalloc((void**) &db_rowptr, sizeof(int) * (shape[0] + 1));
  csr_view<float, int> b(nullptr, db_rowptr, nullptr, shape, 0);

  filter_handle_t handle(allocator);
  filter_compute(handle, a, b, keep_valid_nonzeros());

  hipMalloc((void**) &db_values, handle.result_nnz() * sizeof(float));
  hipMalloc((void**) &db_colind, handle.result_nnz() * sizeof(int));
  std::cout << "result_nnz: " << handle.result_nnz() << std::endl;
  std::span<int> b_rowptr_span(db_rowptr, shape[0] + 1);
  std::span<int> b_colind_span(db_colind, handle.result_nnz());
  std::span<float> b_values_span(db_values, handle.result_nnz());
  b.update(b_values_span, b_rowptr_span, b_colind_span, shape,
           static_cast<int>(handle.result_nnz()));
  filter_execute(handle, a, b, keep_valid_nonzeros());

  std::vector<int> b_rowptr(shape[0] + 1);
  std::vector<int> b_colind(handle.result_nnz());
  std::vector<float> b_values(handle.result_nnz());
  hipMemcpy(b_rowptr.data(), db_rowptr, sizeof(int) * (shape[0] + 1),
             hipMemcpyDeviceToHost);
  hipMemcpy(b_colind.data(), db_colind, sizeof(int) * handle.result_nnz(),
             hipMemcpyDeviceToHost);
  hipMemcpy(b_values.data(), db_values, sizeof(float) * handle.result_nnz(),
             hipMemcpyDeviceToHost);

  /**
   * answer should be
   *  3  0 0 0
   *  0  1 0 0
   *  0 -3 2 0
   * -5  0 0 4
   */
  for (const auto& v : b_rowptr) {
    std::cout << v << " ";
  }
  std::cout << std::endl;
  for (const auto& v : b_colind) {
    std::cout << v << " ";
  }
  std::cout << std::endl;
  for (const auto& v : b_values) {
    std::cout << v << " ";
  }
  std::cout << std::endl;

  hipFree(da_values);
  hipFree(da_rowptr);
  hipFree(da_colind);
  hipFree(db_values);
  hipFree(db_rowptr);
  hipFree(db_colind);
  return 0;
}
