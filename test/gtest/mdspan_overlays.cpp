#include <gtest/gtest.h>

#include "util.hpp"
#include <spblas/spblas.hpp>

// Accessing the data inside mdspan differs between different mdspan
// implementations. The portable way is quite heavy and the following helper
// makes the tests themselves easier to read.
template <class T>
decltype(auto) md_at(T& m, typename T::index_type i, typename T::index_type j) {
#if defined(__cpp_multidimensional_subscript)
  return m[i, j];
#else
  return m(i, j);
#endif
}

TEST(Mdspan, positive_row_major) {
  using T = float;
  using I = spblas::index_t;

  for (auto m : {1, 2, 4}) {
    for (auto n : {1, 2, 4}) {
      auto [b_values, b_shape] = spblas::generate_dense<T>(m, n);
      spblas::mdspan_row_major<T, I> b(b_values.data(), m, n);

      // Traverse by row in inner loop to immitade row-major
      T* tmp = b_values.data();
      for (I i = 0; i < m; ++i) {
        for (I j = 0; j < n; ++j) {
          EXPECT_EQ(md_at(b, i, j), *(tmp++));
        }
      }
    }
  }
}

TEST(Mdspan, postive_col_major) {
  using T = double;
  using I = spblas::index_t;

  for (auto m : {1, 2, 4}) {
    for (auto n : {1, 2, 4}) {
      auto [b_values, b_shape] = spblas::generate_dense<T>(m, n);
      spblas::mdspan_col_major<T, I> b(b_values.data(), m, n);

      // Traverse by column in inner loop to immitade col-major
      T* tmp = b_values.data();
      for (I j = 0; j < n; ++j) {
        for (I i = 0; i < m; ++i) {
          EXPECT_EQ(md_at(b, i, j), *(tmp++));
        }
      }
    }
  }
}

TEST(Mdspan, negative_row_major) {
  using T = double;
  using I = int32_t;

  for (auto [m, n] : {std::pair{2, 4}, std::pair{4, 2}}) {

    auto [b_values, b_shape] = spblas::generate_dense<T>(m, n);
    spblas::mdspan_row_major<T, I> b(b_values.data(), m, n);

    // Traverse by column in inner loop to not immitade row-major
    T* tmp = b_values.data();
    for (I j = 0; j < n; ++j) {
      for (I i = 0; i < m; ++i) {
        // Skip first and last element
        if ((i == 0 && j == 0) || (i == m - 1 && j == n - 1)) {
          tmp++;
          continue;
        }
        EXPECT_NE(md_at(b, i, j), *(tmp++));
      }
    }
  }
}

TEST(Mdspan, negative_col_major) {
  using T = int32_t;
  using I = int64_t;

  for (auto [m, n] : {std::pair{2, 4}, std::pair{4, 2}}) {

    auto [b_values, b_shape] = spblas::generate_dense<T>(m, n);
    spblas::mdspan_col_major<T, I> b(b_values.data(), m, n);

    // Traverse by row in inner loop to not immitade col-major
    T* tmp = b_values.data();
    for (I i = 0; i < m; ++i) {
      for (I j = 0; j < n; ++j) {
        // Skip first and last element
        if ((i == 0 && j == 0) || (i == m - 1 && j == n - 1)) {
          tmp++;
          continue;
        }
        EXPECT_NE(md_at(b, i, j), *(tmp++));
      }
    }
  }
}
