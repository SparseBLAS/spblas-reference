#include <iostream>
#include <spblas/spblas.hpp>

#include <spblas/backend/sycl/spmm_impl.hpp>

#include <thrust/device_vector.h>

#include <cassert>

#include <cmath>
#include <fmt/core.h>
#include <fmt/ranges.h>

int main(int argc, char** argv) {
  using value_t = float;
  using index_t = int32_t;
  using offset_t = int32_t;
  namespace md = spblas::__mdspan;

  offset_t nnz_row = 100;

  index_t m = 100000;
  index_t n = 1;
  index_t k = 100000;

  char method = 'k';

  std::size_t wg_size = 32;

  if (argc >= 2) {
    m = std::atoll(argv[1]);
  }

  if (argc >= 3) {
    k = std::atoll(argv[2]);
  }

  if (argc >= 4) {
    n = std::atoll(argv[3]);
  }

  if (argc >= 5) {
    nnz_row = std::atoll(argv[4]);
  }

  if (argc >= 6) {
    method = argv[5][0];
  }

  if (argc >= 7) {
    wg_size = std::atoll(argv[6]);
  }

  assert(method == 'k' || method == 'j');

  fmt::print("Multiplying {} x {} matrix with {} nnz/row by {} columns.\n", m,
             k, nnz_row, n);
  fmt::print("Using method {} with WG size {}\n", method, wg_size);

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

  if (method == 'k') {
    spblas::spmm_wgsplitk(q, a, b, c, wg_size);
  } else {
    spblas::spmm_wgsplitj(q, a, b, c, wg_size);
  }

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
    fmt::print("OK!\n");
  } else {
    fmt::print("Error!\n");
    return 1;
  }

  // Warmup: call `SpMM` repeatedly for at least 2 seconds.

  double min_warmup_duration = 2;
  auto warmup_begin = std::chrono::high_resolution_clock::now();
  auto warmup_end = warmup_begin;

  while (std::chrono::duration<double>(warmup_end - warmup_begin).count() <
         min_warmup_duration) {
    if (method == 'k') {
      spblas::spmm_wgsplitk(q, a, b, c, wg_size);
    } else {
      spblas::spmm_wgsplitj(q, a, b, c, wg_size);
    }
    warmup_end = std::chrono::high_resolution_clock::now();
  }

  double gb = 1e-9 * (nnz * sizeof(value_t) + nnz * sizeof(index_t) +
                      (m + 1) * sizeof(offset_t) + k * n * sizeof(value_t) +
                      m * n * sizeof(value_t));

  double gflops = 1e-9 * 2 * nnz * n;

  double max_bw = 456;

  std::size_t n_iterations = 10;

  std::vector<double> durations;
  durations.reserve(n_iterations);

  for (std::size_t i = 0; i < n_iterations; i++) {
    auto begin = std::chrono::high_resolution_clock::now();
    if (method == 'k') {
      spblas::spmm_wgsplitk(q, a, b, c, wg_size);
    } else {
      spblas::spmm_wgsplitj(q, a, b, c, wg_size);
    }
    auto end = std::chrono::high_resolution_clock::now();
    double duration = std::chrono::duration<double>(end - begin).count();
    double gb_s = gb / duration;
    double gflops_s = gflops / duration;

    fmt::print("Completed in {} s (achieved {} GB/s)\n", duration, gb_s);
    fmt::print("Achieved {} GFLOPs\n", gflops_s);

    durations.push_back(duration);
  }

  fmt::print("Durations: {}\n", durations);

  std::sort(durations.begin(), durations.end());

  double median_duration = durations[durations.size() / 2];

  double median_gb_s = gb / median_duration;
  double median_gflops_s = gflops / median_duration;

  fmt::print("Median duration {} ({} GB/s) {}% of peak.\n", median_duration,
             median_gb_s, 100 * (median_gb_s / max_bw));
  fmt::print("Median achieved {} GFLOPs\n", median_gflops_s);

  return 0;
}
