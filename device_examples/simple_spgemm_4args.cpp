#include <iostream>

#include <fmt/core.h>
#include <fmt/ranges.h>
#include <iomanip>
#include <spblas/spblas.hpp>

class amd_allocator : public spblas::allocator {
public:
  void alloc(void** ptrptr, size_t size) const override {
    hipMalloc(ptrptr, size);
  }

  void free(void* ptr) const override {
    hipFree(ptr);
  }
};

int main(int argc, char** argv) {
  using namespace spblas;
  namespace md = spblas::__mdspan;

  spblas::index_t m = 100;
  spblas::index_t n = 10;
  spblas::index_t k = 100;
  spblas::index_t nnz = 150;

  auto allocator = std::make_shared<amd_allocator>();
  spblas::spgemm_handle_t spgemm_handle(allocator);

  auto&& [a_values, a_rowptr, a_colind, a_shape, as] =
      generate_csr<float, int>(m, k, nnz);
  auto&& [b_values, b_rowptr, b_colind, b_shape, bs] =
      generate_csr<float, int>(k, n, nnz);
  auto&& [d_values, d_rowptr, d_colind, d_shape, ds] =
      generate_csr<float, int>(m, n, nnz);
std::cout << as << " " << bs << " " << ds << std::endl;
  // init to one;
  for (auto& val : a_values) {
    val = 1.0;
  }
  for (auto& val : b_values) {
    val = 1.0;
  }
  for (auto& val : d_values) {
    val = 100.0;
  }
  for (auto& val: d_rowptr) {
    std::cout << val << " ";
  }
  std::cout << std::endl;
  for (auto& val: d_colind) {
    std::cout << val << " ";
  }
  std::cout << std::endl;
  float *da_values, *db_values, *dd_values;
  int *da_rowptr, *da_colind, *db_rowptr, *db_colind, *dd_rowptr, *dd_colind;

  hipMalloc((void**) &da_values, sizeof(float) * nnz);
  hipMemcpy((void**) da_values, a_values.data(), sizeof(float) * nnz,
            hipMemcpyHostToDevice);
  std::span<float> a_values_span(da_values, nnz);
  hipMalloc((void**) &da_rowptr, sizeof(int) * (m + 1));
  hipMemcpy(da_rowptr, a_rowptr.data(), sizeof(int) * (m + 1),
            hipMemcpyHostToDevice);
  std::span<int> a_rowptr_span(da_rowptr, m + 1);
  hipMalloc((void**) &da_colind, sizeof(int) * nnz);
  hipMemcpy(da_colind, a_colind.data(), sizeof(int) * nnz,
            hipMemcpyHostToDevice);
  std::span<int> a_colind_span(da_colind, nnz);
  csr_view<float, int> a(a_values_span, a_rowptr_span, a_colind_span, a_shape,
                         nnz);

  hipMalloc((void**) &db_values, sizeof(float) * nnz);
  hipMemcpy(db_values, b_values.data(), sizeof(float) * nnz,
            hipMemcpyHostToDevice);
  std::span<float> b_values_span(db_values, nnz);
  hipMalloc((void**) &db_rowptr, sizeof(int) * (k + 1));
  hipMemcpy(db_rowptr, b_rowptr.data(), sizeof(int) * (k + 1),
            hipMemcpyHostToDevice);
  std::span<int> b_rowptr_span(db_rowptr, k + 1);
  hipMalloc((void**) &db_colind, sizeof(int) * nnz);
  hipMemcpy(db_colind, b_colind.data(), sizeof(int) * nnz,
            hipMemcpyHostToDevice);
  std::span<int> b_colind_span(db_colind, nnz);
  csr_view<float, int> b(b_values_span, b_rowptr_span, b_colind_span, b_shape,
                         nnz);

  hipMalloc((void**) &dd_values, sizeof(float) * nnz);
  hipMemcpy(dd_values, d_values.data(), sizeof(float) * nnz,
            hipMemcpyHostToDevice);
  std::span<float> d_values_span(dd_values, nnz);
  hipMalloc((void**) &dd_rowptr, sizeof(int) * (m + 1));
  hipMemcpy(dd_rowptr, d_rowptr.data(), sizeof(int) * (m + 1),
            hipMemcpyHostToDevice);
  std::span<int> d_rowptr_span(dd_rowptr, m + 1);
  hipMalloc((void**) &dd_colind, sizeof(int) * nnz);
  hipMemcpy(dd_colind, d_colind.data(), sizeof(int) * nnz,
            hipMemcpyHostToDevice);
  std::span<int> d_colind_span(dd_colind, nnz);
  csr_view<float, int> d(d_values_span, d_rowptr_span, d_colind_span, d_shape,
                         nnz);

  int* dc_rowptr;
  hipMalloc((void**) &dc_rowptr, sizeof(int) * (m + 1));
  // std::span<int> c_rowptr_span(dc_rowptr, m+1);

  csr_view<float, int> c(nullptr, dc_rowptr, nullptr, {m, n}, 0);
  multiply_symbolic_compute(spgemm_handle, a, b, c, scaled(1.0, d));

  std::cout << "nnz" << spgemm_handle.result_nnz() << std::endl;
  float* dc_values;
  int* dc_colind;
  hipMalloc((void**) &dc_values, spgemm_handle.result_nnz() * sizeof(float));
  hipMalloc((void**) &dc_colind, spgemm_handle.result_nnz() * sizeof(int));

  std::span<int> c_rowptr_span(dc_rowptr, m + 1);
  std::span<int> c_colind_span(dc_colind, spgemm_handle.result_nnz());
  std::span<float> c_values_span(dc_values, spgemm_handle.result_nnz());
  c.update(c_values_span, c_rowptr_span, c_colind_span, {m, n},
           (int) spgemm_handle.result_nnz());
//   multiply_execute(spgemm_handle, a, b, c, scaled(1.0, d));
  multiply_symbolic_fill(spgemm_handle, a, b, c, scaled(1.0, d));
//   std::cout << "symbolic finish" << std::endl;
  std::vector<float> c_values(spgemm_handle.result_nnz());
  for (int i = 0; i < 5; i++) {
    std::cout << i << " step: ";
    multiply_numeric(spgemm_handle, a, b, c, scaled(1.0, d));
    // std::cout << "??" << std::endl;
    hipMemcpy(c_values.data(), dc_values,
              sizeof(float) * spgemm_handle.result_nnz(),
              hipMemcpyDeviceToHost);
    for (const auto& val : c_values) {
      std::cout << std::setw(2) << val << " ";
    }
    std::cout << std::endl;

    // change the A value;
    for (auto& val : a_values) {
      val = i + 2;
    }
    hipMemcpy((void**) da_values, a_values.data(), sizeof(float) * nnz,
              hipMemcpyHostToDevice);
  }
  hipFree(da_values);
  hipFree(da_rowptr);
  hipFree(da_colind);
  hipFree(db_values);
  hipFree(db_rowptr);
  hipFree(db_colind);
  hipFree(dc_rowptr);
  hipFree(dc_values);
  hipFree(dc_colind);
  hipFree(dd_rowptr);
  hipFree(dd_values);
  hipFree(dd_colind);

  return 0;
}
