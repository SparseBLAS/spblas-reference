#include <iostream>
#include <spblas/spblas.hpp>

#include <spblas/backend/sycl/spmm_impl.hpp>

#include <thrust/device_vector.h>

#include <cmath>
#include <fmt/core.h>
#include <fmt/ranges.h>

int main(int argc, char** argv) {
  using value_t = float;
  using index_t = spblas::index_t;
  using offset_t = spblas::offset_t;
  namespace md = spblas::__mdspan;

  offset_t nnz_row = 1000;

  index_t m = 10000;
  index_t n = 1;
  index_t k = 10000;
  offset_t nnz_in = m * nnz_row;

  auto&& [values, rowptr, colind, shape, nnz] =
      spblas::generate_csr<value_t, index_t, offset_t>(m, k, nnz_in);

  thrust::device_vector<value_t> d_values(values);
  thrust::device_vector<offset_t> d_rowptr(rowptr);
  thrust::device_vector<index_t> d_colind(colind);

  spblas::csr_view<value_t, index_t, offset_t> a(
      d_values.data().get(), d_rowptr.data().get(), d_colind.data().get(),
      shape, nnz);

  std::vector<value_t> b_values(k * n, 1);
  std::vector<value_t> c_values(m * n, 0);

  thrust::device_vector<value_t> d_b(b_values);
  thrust::device_vector<value_t> d_c(c_values);

  md::mdspan b(d_b.data().get(), k, n);
  md::mdspan c(d_c.data().get(), m, n);

  sycl::queue q(sycl::gpu_selector_v);

  spblas::spmm(q, a, b, c);

  thrust::copy(d_c.begin(), d_c.end(), c_values.begin());

  std::vector<value_t> c_ref(m * n, 0);

  spblas::csr_view<value_t, index_t, offset_t> a_view(
      values.data(), rowptr.data(), colind.data(), shape, nnz);
  md::mdspan b_view(b_values.data(), k, n);
  md::mdspan c_view(c_ref.data(), m, n);

  spblas::multiply(a_view, b_view, c_view);

  // Compare results
  const float epsilon = 64 * std::numeric_limits<float>::epsilon();
  const float abs_th = std::numeric_limits<float>::min();
  bool results_match = true;

  for (std::size_t i = 0; i < c_ref.size(); ++i) {
    float diff = std::abs(c_ref[i] - c_values[i]);
    float norm = std::min(std::abs(c_ref[i]) + std::abs(c_values[i]),
                          std::numeric_limits<float>::max());
    float abs_error = std::max(abs_th, epsilon * norm);

    if (diff > abs_error) {
      results_match = false;
      std::cout << "Mismatch at index " << i << ": "
                << "SYCL result = " << c_values[i]
                << ", Reference = " << c_ref[i] << "\n";
      break;
    }
  }

  if (results_match) {
    std::cout << "SYCL and reference results match!\n";
  }

  std::size_t n_iterations = 10;

  for (std::size_t i = 0; i < n_iterations; i++) {
    auto begin = std::chrono::high_resolution_clock::now();
    spblas::spmm(q, a, b, c);
    auto end = std::chrono::high_resolution_clock::now();
    double duration = std::chrono::duration<double>(end - begin).count();
    double gb = 1e-9 * (nnz * sizeof(value_t) + nnz * sizeof(index_t) +
                        (m + 1) * sizeof(offset_t) + k * n * sizeof(value_t) +
                        m * n * sizeof(value_t));
    double gb_s = gb / duration;

    fmt::print("Completed in {} s (achieved {} GB/s)\n", duration, gb_s);
  }

  return 0;
}
