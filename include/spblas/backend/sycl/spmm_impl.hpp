#pragma once

#include <spblas/backend/concepts.hpp>
#include <spblas/backend/cpos.hpp>
#include <spblas/detail/log.hpp>
#include <spblas/detail/ranges.hpp>
#include <spblas/detail/view_inspectors.hpp>
#include <sycl/sycl.hpp>

#include <fmt/core.h>

namespace spblas {

template <matrix A, matrix B, matrix C>
  requires(__backend::row_iterable<A> && __backend::lookupable<B> &&
           __backend::lookupable<C>)
void spmm(sycl::queue& q, A&& a, B&& b, C&& c) {
  log_trace("");

  std::size_t wg_size = 32;
  std::size_t n_workgroups = __backend::shape(a)[0];

  fmt::print(stderr, "Launching parallel_for...\n");

  q.parallel_for(sycl::nd_range<1>{wg_size * n_workgroups, wg_size},
                 [=](auto nd_idx) {
                   auto gid = nd_idx.get_group(0);
                   auto lid = nd_idx.get_local_id(0);
                   auto lsz = nd_idx.get_local_range(0);

                   auto i = gid;

                   if (i < __backend::shape(a)[0]) {
                     auto row = __backend::lookup_row(a, i);

                     for (auto elem_idx = lid; elem_idx < row.size();
                          elem_idx += lsz) {
                       auto&& [k, a_v] = row[elem_idx];

                       for (int j = 0; j < __backend::shape(c)[1]; j++) {
                         auto local_product = a_v * __backend::lookup(b, k, j);
                         auto group_sum = sycl::reduce_over_group(
                             nd_idx.get_group(), local_product, sycl::plus<>());
                         if (lid == 0) {
                           __backend::lookup(c, i, j) += group_sum;
                           /*
                           sycl::atomic_ref<tensor_scalar_t<C>,
                           sycl::memory_order::relaxed,
                                         sycl::memory_scope::device,
                                         sycl::access::address_space::global_space>
                               c_ref(__backend::lookup(c, i, j));
                               */
                           // c_ref += group_sum;
                         }
                       }
                     }
                   }
                 })
      .wait();

  fmt::print(stderr, "Returning from SpMM...\n");
}

} // namespace spblas
