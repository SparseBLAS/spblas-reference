#include <gtest/gtest.h>

#include "util.hpp"
#include <spblas/backend/spa_accumulator.hpp>
#include <spblas/spblas.hpp>

#include <fmt/core.h>
#include <fmt/ranges.h>

TEST(CsrView, Add_CSR_CSR_CSR) {
  using T = float;
  using I = spblas::index_t;

  for (auto&& [m, n, nnz] : util::dims) {
    auto [a_values, a_rowptr, a_colind, a_shape, a_nnz] =
        spblas::generate_csr<T, I>(m, n, nnz);

    auto [b_values, b_rowptr, b_colind, b_shape, b_nnz] =
        spblas::generate_csr<T, I>(m, n, nnz);

    spblas::csr_view<T, I> a(a_values, a_rowptr, a_colind, a_shape, a_nnz);
    spblas::csr_view<T, I> b(b_values, b_rowptr, b_colind, b_shape, b_nnz);

    std::vector<I> c_rowptr(m + 1);

    spblas::csr_view<T, I> c(nullptr, c_rowptr.data(), nullptr, {m, n}, 0);

    auto info = spblas::add_inspect(a, b, c);

    std::vector<T> c_values(info.result_nnz());
    std::vector<I> c_colind(info.result_nnz());

    c.update(c_values, c_rowptr, c_colind);

    spblas::add_compute(info, a, b, c);

    spblas::__backend::spa_accumulator<T, I> c_row_ref(
        spblas::__backend::shape(c)[1]);

    for (I i = 0; i < spblas::__backend::shape(c)[0]; i++) {
      c_row_ref.clear();

      for (auto&& [j, v] : spblas::__backend::lookup_row(a, i)) {
        c_row_ref[j] += v;
      }

      for (auto&& [j, v] : spblas::__backend::lookup_row(b, i)) {
        c_row_ref[j] += v;
      }

      auto&& c_row = spblas::__backend::lookup_row(c, i);

      for (auto&& [j, v] : c_row) {
        EXPECT_EQ_(c_row_ref[j], v);
      }
    }
  }
}
