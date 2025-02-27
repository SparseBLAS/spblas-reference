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
  int a_nnz = 6;
  std::vector<float> a_values{3, 1, -3, 2, -5, 4};
  std::vector<int> a_rowptr{0, 1, 2, 4, 6};
  std::vector<int> a_colind{0, 1, 1, 2, 0, 3};
  spblas::index<int> shape(4, 4);

  /**
   *  3  0 0 1
   *  0  1 0 0
   * -2  0 2 0
   *  0  0 0 4
   */
  int b_nnz = 6;
  std::vector<float> b_values{3, 1, 1, -2, 2, 4};
  std::vector<int> b_rowptr{0, 2, 3, 5, 6};
  std::vector<int> b_colind{0, 3, 1, 0, 2, 3};

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

  hipMalloc((void**) &db_values, sizeof(float) * b_nnz);
  hipMalloc((void**) &db_rowptr, sizeof(int) * (shape[0] + 1));
  hipMalloc((void**) &db_colind, sizeof(int) * b_nnz);
  hipMemcpy(db_values, b_values.data(), sizeof(float) * b_nnz,
             hipMemcpyHostToDevice);
  hipMemcpy(db_rowptr, b_rowptr.data(), sizeof(int) * (shape[0] + 1),
             hipMemcpyHostToDevice);
  hipMemcpy(db_colind, b_colind.data(), sizeof(int) * b_nnz,
             hipMemcpyHostToDevice);
  csr_view<float, int> b(db_values, db_rowptr, db_colind, shape, b_nnz);

  int* dc_rowptr;
  hipMalloc((void**) &dc_rowptr, sizeof(int) * (shape[0] + 1));

  csr_view<float, int> c(nullptr, dc_rowptr, nullptr, shape, 0);

  add_handle_t handle(allocator);
  std::cout << "QQ" << std::endl;
  add_compute(handle, a, b, c);

  float* dc_values;
  int* dc_colind;
  hipMalloc((void**) &dc_values, handle.result_nnz() * sizeof(float));
  hipMalloc((void**) &dc_colind, handle.result_nnz() * sizeof(int));
  std::cout << "result_nnz: " << handle.result_nnz() << std::endl;
  std::span<int> c_rowptr_span(dc_rowptr, shape[0] + 1);
  std::span<int> c_colind_span(dc_colind, handle.result_nnz());
  std::span<float> c_values_span(dc_values, handle.result_nnz());
  c.update(c_values_span, c_rowptr_span, c_colind_span, shape,
           static_cast<int>(handle.result_nnz()));
  add_execute(handle, a, b, c);

  std::vector<int> c_rowptr(shape[0] + 1);
  std::vector<int> c_colind(handle.result_nnz());
  std::vector<float> c_values(handle.result_nnz());
  hipMemcpy(c_rowptr.data(), dc_rowptr, sizeof(int) * (shape[0] + 1),
             hipMemcpyDeviceToHost);
  hipMemcpy(c_colind.data(), dc_colind, sizeof(int) * handle.result_nnz(),
             hipMemcpyDeviceToHost);
  hipMemcpy(c_values.data(), dc_values, sizeof(float) * handle.result_nnz(),
             hipMemcpyDeviceToHost);

  /**
   * answer should be
   *  6  0 0 1
   *  0  2 0 0
   * -2 -3 4 0
   * -5  0 0 8
   */
  for (const auto& v : c_rowptr) {
    std::cout << v << " ";
  }
  std::cout << std::endl;
  for (const auto& v : c_colind) {
    std::cout << v << " ";
  }
  std::cout << std::endl;
  for (const auto& v : c_values) {
    std::cout << v << " ";
  }
  std::cout << std::endl;

  hipFree(da_values);
  hipFree(da_rowptr);
  hipFree(da_colind);
  hipFree(db_values);
  hipFree(db_rowptr);
  hipFree(db_colind);
  hipFree(dc_values);
  hipFree(dc_rowptr);
  hipFree(dc_colind);
  return 0;
}
