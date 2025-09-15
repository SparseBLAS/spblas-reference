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
void spmm_wgsplitk(sycl::queue& q, A&& a, B&& b, C&& c,
                   std::size_t wg_size = 64) {
  log_trace("");

  std::size_t n_workgroups = __backend::shape(a)[0];

  q.parallel_for<class SpMMSplitK>(
       sycl::nd_range<1>{wg_size * n_workgroups, wg_size},
       [=](auto nd_idx) {
         auto gid = nd_idx.get_group(0);
         auto lid = nd_idx.get_local_id(0);
         auto lsz = nd_idx.get_local_range(0);

         auto i = gid;

         if (i < __backend::shape(a)[0]) {
           auto row = __backend::lookup_row(a, i);

           for (auto elem_idx = lid; elem_idx < row.size(); elem_idx += lsz) {
             auto&& [k, a_v] = row[elem_idx];

             for (int j = 0; j < __backend::shape(c)[1]; j++) {
               auto local_product = a_v * __backend::lookup(b, k, j);
               auto group_sum = sycl::reduce_over_group(
                   nd_idx.get_group(), local_product, sycl::plus<>());
               if (lid == 0) {
                 __backend::lookup(c, i, j) += group_sum;
               }
             }
           }
         }
       })
      .wait();
}

template <matrix A, matrix B, matrix C>
  requires(__backend::row_iterable<A> && __backend::lookupable<B> &&
           __backend::lookupable<C>)
void spmm_wgsplitj(sycl::queue& q, A&& a, B&& b, C&& c,
                   std::size_t wg_size = 64) {
  log_trace("");

  std::size_t n_workgroups = __backend::shape(a)[0];

  q.parallel_for<class SpMMSplitJ>(
       sycl::nd_range<1>{wg_size * n_workgroups, wg_size},
       [=](auto nd_idx) {
         auto gid = nd_idx.get_group(0);
         auto lid = nd_idx.get_local_id(0);
         auto lsz = nd_idx.get_local_range(0);

         auto i = gid;

         if (i < __backend::shape(a)[0]) {
           auto row = __backend::lookup_row(a, i);

           for (auto&& [k, a_v] : row) {
             for (auto j = lid; j < __backend::shape(c)[1]; j += lsz) {
               __backend::lookup(c, i, j) += a_v * __backend::lookup(b, k, j);
             }
           }
         }
       })
      .wait();
}

} // namespace spblas
