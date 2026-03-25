#include <gtest/gtest.h>

#include "util.hpp"
#include <spblas/spblas.hpp>

TEST(Mdspan, positive_row_major) {
  using T = float;
  using I = spblas::index_t;

  for (auto m : {1, 2, 4}) {
    for (auto n : {1, 2, 4}) {

      auto [b_values, b_shape] = spblas::generate_dense<T>(m, n);
      spblas::mdspan_row_major<I, T> b(b_values.data(), m, n);


      // Traverse by row in inner loop to immitade row-major
      T* tmp = b_values.data();
      for (I i = 0; i < m; ++i) {
        for (I j = 0; j < n; ++j) {
            EXPECT_EQ((b[i,j]), *(tmp++));
        }
      }
    }
  }
}

TEST(Mdspan, postive_col_major) {
  using T = float;
  using I = spblas::index_t;

  for (auto m : {1, 2, 4}) {
    for (auto n : {1, 2, 4}) {

      auto [b_values, b_shape] = spblas::generate_dense<T>(m, n);
      spblas::mdspan_col_major<I, T> b(b_values.data(), m, n);

      // Traverse by column in inner loop to immitade col-major
      T* tmp = b_values.data();
      for (I j = 0; j < n; ++j) {
        for (I i = 0; i < m; ++i) {
          EXPECT_EQ((b[i,j]), *(tmp++));
        }
      }
    }
  }
}

TEST(Mdspan, negative_row_major) {
  using T = float;
  using I = spblas::index_t;

  for (auto [m, n] : {std::pair{2, 4}, std::pair{4, 2}}) {

    auto [b_values, b_shape] = spblas::generate_dense<T>(m, n);
    spblas::mdspan_row_major<I, T> b(b_values.data(), m, n);

    // Traverse by column in inner loop to not immitade row-major
    T* tmp = b_values.data();
    for (I j = 0; j < n; ++j) {
      for (I i = 0; i < m; ++i) {
        // Skip first and last element
        if ((i == 0 && j == 0) || (i == m - 1 && j == n - 1)) {
          tmp++;
          continue;
        }
        EXPECT_NE((b[i, j]), *(tmp++));
      }
    }
  }
}

TEST(Mdspan, negative_col_major) {
  using T = float;
  using I = spblas::index_t;

  for (auto [m, n] : {std::pair{2, 4}, std::pair{4, 2}}) {

    auto [b_values, b_shape] = spblas::generate_dense<T>(m, n);
    spblas::mdspan_col_major<I, T> b(b_values.data(), m, n);

    // Traverse by row in inner loop to not immitade col-major
    T* tmp = b_values.data();
    for (I i = 0; i < m; ++i) {
      for (I j = 0; j < n; ++j) {
        // Skip first and last element
        if ((i == 0 && j == 0) || (i == m - 1 && j == n - 1)) {
          tmp++;
          continue;
        }
        EXPECT_NE((b[i, j]), *(tmp++));
      }
    }
  }
}
