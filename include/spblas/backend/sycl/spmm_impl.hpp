#pragma once

#include <spblas/backend/concepts.hpp>
#include <spblas/backend/cpos.hpp>
#include <spblas/detail/log.hpp>
#include <spblas/detail/ranges.hpp>
#include <spblas/detail/view_inspectors.hpp>
#include <sycl/sycl.hpp>

namespace spblas {

template <matrix A, matrix B, matrix C>
  requires(__backend::row_iterable<A> && __backend::lookupable<B> &&
           __backend::lookupable<C>)
void spmm(sycl::queue& q, A&& a, B&& b, C&& c) {
  log_trace("");

  // Get base matrices and shapes
  auto a_base = __detail::get_ultimate_base(a);
  auto b_base = __detail::get_ultimate_base(b);

  const auto m = __backend::shape(a_base)[0]; // Number of rows in A
  const auto k = __backend::shape(a_base)[1]; // Number of cols in A
  const auto n = __backend::shape(b_base)[1]; // Number of cols in B

  // Get scaling factor if any
  auto alpha_optional = __detail::get_scaling_factor(a, b);
  auto alpha = alpha_optional.value_or(1);

  // Define workgroup size and compute grid dimensions
  constexpr size_t WG_SIZE = 256;
  constexpr size_t ITEMS_PER_THREAD = 4;

  // Launch kernel with workgroups
  q.submit([&](sycl::handler& h) {
    h.parallel_for(
      sycl::nd_range<1>{
        sycl::range<1>{((m + WG_SIZE - 1) / WG_SIZE) * WG_SIZE},
        sycl::range<1>{WG_SIZE}
      },
      [=](sycl::nd_item<1> item) {
      const auto row_idx = item.get_global_id(0);
      const auto local_idx = item.get_local_id(0);
      auto group = item.get_group();

      // Skip if this thread is beyond matrix dimensions
      if (row_idx >= m)
        return;

      // Get the row using customization point
      auto row = __backend::lookup_row(a_base, row_idx);

      // Process columns in chunks
      for (size_t j_base = 0; j_base < n; j_base += WG_SIZE) {
        // Each thread processes ITEMS_PER_THREAD columns
        tensor_scalar_t<C> local_sums[ITEMS_PER_THREAD] = {};

        // Iterate over non-zeros in the row using customization point
        for (auto&& [col, val] : row) {
          // Load B values for this column into local memory
          auto b_row = &b_base.data_handle()[col * n + j_base];

// Compute partial sums for this thread's columns
#pragma unroll
          for (size_t j_offset = 0; j_offset < ITEMS_PER_THREAD; j_offset++) {
            const size_t j = j_base + local_idx + j_offset * WG_SIZE;
            if (j < n) {
              local_sums[j_offset] +=
                  alpha * val * b_row[local_idx + j_offset * WG_SIZE];
            }
          }
        }

// Store results
#pragma unroll
        for (size_t j_offset = 0; j_offset < ITEMS_PER_THREAD; j_offset++) {
          const size_t j = j_base + local_idx + j_offset * WG_SIZE;
          if (j < n)
            cal_sums[ITEMS_PER_THREAD] = {};

          // Iterate over non-zeros in the row using customization point
          for (auto&& [col, val] : row) {
            // Load B values for this column into local memory
            auto b_row = &b_base.data_handle()[col * n + j_base];

// Compute partial sums for this thread's columns
#pragma unroll
            for (size_t j_offset = 0; j_offset < ITEMS_PER_THREAD; j_offset++) {
              const size_t j = j_base + local_idx + j_offset * WG_SIZE;
              if (j < n) {
                local_sums[j_offset] +=
                    alpha * val * b_row[local_idx + j_offset * WG_SIZE];
              }
            }
          }

// Store results
#pragma unroll
          for (size_t j_offset = 0; j_offset < ITEMS_PER_THREAD; j_offset++) {
            const size_t j = j_base + local_idx + j_offset * WG_SIZE;
            if (j < n) {
              {
                c.data_handle()[row_idx * n + j] = local_sums[j_offset];
              }
            }
          }
        }
    );
      }).wait();
}

}
