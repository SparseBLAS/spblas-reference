#pragma once

#include <spblas/backend/concepts.hpp>
#include <spblas/backend/cpos.hpp>
#include <spblas/detail/log.hpp>
#include <spblas/detail/ranges.hpp>
#include <spblas/detail/view_inspectors.hpp>
#include <sycl/sycl.hpp>

#include <fmt/core.h>

namespace spblas {

// Optimizations:
// - Move reduction out of inner loop
// - LLC optimization (block __backend::shape(c)[1] into multiple kernels)
//    * Copy optimization on B?
//    * Transpose B?
// - L1 optimization: copy b to shared memory

template <matrix A, matrix B, matrix C>
  requires(__backend::row_iterable<A> && __backend::lookupable<B> &&
           __backend::lookupable<C>)
void spmm_wgsplitk(sycl::queue& q, A&& a, B&& b, C&& c,
                   std::size_t wg_size = 64, std::size_t n_workgroups = 0) {
  log_trace("");

  if (n_workgroups == 0) {
    n_workgroups = __backend::shape(a)[0];
  }

  q.parallel_for<class SpMMSplitK>(
       sycl::nd_range<1>{wg_size * n_workgroups, wg_size},
       [=](auto nd_idx) {
         auto gid = nd_idx.get_group(0);
         auto lid = nd_idx.get_local_id(0);
         auto lsz = nd_idx.get_local_range(0);

         for (auto i = gid; i < __backend::shape(a)[0]; i += n_workgroups) {
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
void spmm_wgsplitk_reorder(sycl::queue& q, A&& a, B&& b, C&& c,
                           std::size_t wg_size = 64,
                           std::size_t n_workgroups = 0) {
  log_trace("");

  if (n_workgroups == 0) {
    n_workgroups = __backend::shape(a)[0];
  }

  q.parallel_for<class SpMMSplitKReorder>(
       sycl::nd_range<1>{wg_size * n_workgroups, wg_size},
       [=](auto nd_idx) {
         auto gid = nd_idx.get_group(0);
         auto lid = nd_idx.get_local_id(0);
         auto lsz = nd_idx.get_local_range(0);

         for (auto i = gid; i < __backend::shape(a)[0]; i += n_workgroups) {
           auto row = __backend::lookup_row(a, i);

           using T = std::remove_cvref_t<decltype(std::get<1>(row[0]) *
                                                  __backend::lookup(b, 0, 0))>;

           for (int j = 0; j < __backend::shape(c)[1]; j++) {
             T local_sum = 0;
             for (auto elem_idx = lid; elem_idx < row.size(); elem_idx += lsz) {
               auto&& [k, a_v] = row[elem_idx];

               auto local_product = a_v * __backend::lookup(b, k, j);
               local_sum += local_product;
             }

             auto group_sum = sycl::reduce_over_group(
                 nd_idx.get_group(), local_sum, sycl::plus<>());
             if (lid == 0) {
               __backend::lookup(c, i, j) += group_sum;
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
                   std::size_t wg_size = 64, std::size_t n_workgroups = 2048) {
  log_trace("");

  if (n_workgroups == 0) {
    n_workgroups = __backend::shape(a)[0];
  }

  // std::size_t j_bs = __backend::shape(c)[1];
  std::size_t j_bs = wg_size;

  auto max_slm_size =
      q.get_device().get_info<sycl::info::device::local_mem_size>();

  q.submit([&](auto&& h) {
     // sycl::local_accessor<char> slm(max_slm_size, h);

     h.template parallel_for<class SpMMSplitJ>(
         sycl::nd_range<1>{wg_size * n_workgroups, wg_size}, [=](auto nd_idx) {
           auto gid = nd_idx.get_group(0);
           auto lid = nd_idx.get_local_id(0);
           auto lsz = nd_idx.get_local_range(0);

           for (int j_block = 0; j_block < __backend::shape(c)[1];
                j_block += j_bs) {

             for (auto i = gid; i < __backend::shape(a)[0]; i += n_workgroups) {
               auto row = __backend::lookup_row(a, i);

               for (auto&& [k, a_v] : row) {
                 for (auto j = j_block + lid;
                      j < std::min(j_block + j_bs, __backend::shape(c)[1]);
                      j += lsz) {
                   __backend::lookup(c, i, j) +=
                       a_v * __backend::lookup(b, k, j);
                 }
               }
             }
           }
         });
   }).wait();
}

} // namespace spblas
