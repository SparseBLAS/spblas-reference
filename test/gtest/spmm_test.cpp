#include <gtest/gtest.h>

#include <spblas/spblas.hpp>

TEST(CsrView, SpMM) {
  namespace md = spblas::__mdspan;

  using T = int;
  using I = int;

  for (auto&& [m, k, nnz] :
       {std::tuple(1000, 100, 100), std::tuple(100, 1000, 10000),
        std::tuple(40, 40, 1000)}) {
    for (auto&& n : {1, 8, 32, 64, 512}) {
      auto [values, rowptr, colind, shape, _] =
          spblas::generate_csr<T, I>(m, k, nnz);

      spblas::csr_view<T, I> a(values, rowptr, colind, shape, nnz);

      auto [b_values, b_shape] = spblas::generate_dense<T>(k, n);

      std::vector<T> c_values(m * n, 0);

      md::mdspan b(b_values.data(), k, n);
      md::mdspan c(c_values.data(), m, n);

      spblas::multiply(a, b, c);

      std::vector<T> c_ref(m * n, 0);

      for (I i = 0; i < m; i++) {
        for (I k_ptr = rowptr[i]; k_ptr < rowptr[i + 1]; k_ptr++) {
          I k = colind[k_ptr];
          T v = values[k_ptr];

          for (I j = 0; j < n; j++) {
            c_ref[i * n + j] += v * b_values[k * n + j];
          }
        }
      }

      for (I i = 0; i < c_ref.size(); i++) {
        EXPECT_EQ(c_ref[i], c_values[i]);
      }
    }
  }
}
