#include <gtest/gtest.h>

#include <spblas/spblas.hpp>

#include <fmt/core.h>
#include <fmt/ranges.h>

TEST(CsrView, SpGEMM) {
  using T = int;
  using I = int;

  for (auto&& [m, k, nnz] :
       {std::tuple(1000, 100, 100), std::tuple(100, 1000, 10000),
        std::tuple(40, 40, 1000)}) {
    for (auto&& n : {m, k}) {
      auto [a_values, a_rowptr, a_colind, a_shape, a_nnz] =
          spblas::generate_csr<T, I>(m, k, nnz);

      auto [b_values, b_rowptr, b_colind, b_shape, b_nnz] =
          spblas::generate_csr<T, I>(k, n, nnz);

      spblas::csr_view<T, I> a(a_values, a_rowptr, a_colind, a_shape, a_nnz);
      spblas::csr_view<T, I> b(b_values, b_rowptr, b_colind, b_shape, b_nnz);

      std::vector<T> c_rowptr(m + 1);

      spblas::csr_view<T, I> c(nullptr, c_rowptr.data(), nullptr, {m, n}, 0);

      auto info = spblas::multiply_inspect(a, b, c);

      std::vector<T> c_values(info.result_nnz());
      std::vector<I> c_colind(info.result_nnz());

      c.update(c_values, c_rowptr, c_colind);

      spblas::multiply_execute(info, a, b, c);

      spblas::__backend::spa_accumulator<T, I> c_row_ref(
          spblas::__backend::shape(c)[1]);

      for (auto&& [i, a_row] : spblas::__backend::rows(a)) {
        c_row_ref.clear();
        for (auto&& [k, a_v] : a_row) {
          auto&& b_row = spblas::__backend::lookup_row(b, k);

          for (auto&& [j, b_v] : b_row) {
            c_row_ref[j] += a_v * b_v;
          }
        }

        auto&& c_row = spblas::__backend::lookup_row(c, i);

        EXPECT_EQ(c_row_ref.size(), c_row.size());

        for (auto&& [j, c_v] : c_row) {
          EXPECT_EQ(c_row_ref[j], c_v);
        }
      }
    }
  }
}
