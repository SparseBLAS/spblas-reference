#pragma once

#include <algorithm>
#include <gtest/gtest.h>
#include <numeric>
#include <random>
#include <tuple>
#include <vector>

#include <spblas/spblas.hpp>

#define EXPECT_EQ_(t, u)                                                       \
  if constexpr (std::floating_point<std::remove_cvref_t<decltype((t))>> ||     \
                std::floating_point<std::remove_cvref_t<decltype((u))>>) {     \
    auto epsilon =                                                             \
        64 *                                                                   \
        std::numeric_limits<std::remove_cvref_t<decltype((t))>>::epsilon();    \
    auto abs_th =                                                              \
        std::numeric_limits<std::remove_cvref_t<decltype((t))>>::min();        \
    auto diff = std::abs((t) - (u));                                           \
    auto norm = std::min(                                                      \
        std::abs((t)) + std::abs((u)),                                         \
        std::numeric_limits<std::remove_cvref_t<decltype((t))>>::max());       \
    auto abs_error = std::max(abs_th, epsilon * norm);                         \
    EXPECT_NEAR((t), (u), abs_error);                                          \
  } else {                                                                     \
    EXPECT_EQ((t), (u));                                                       \
  }

namespace util {

inline auto dims =
    std::vector({std::tuple(1000, 100, 100), std::tuple(100, 1000, 10000),
                 std::tuple(40, 40, 1000)});

inline auto square_dims =
    std::vector({std::tuple(1000, 1000, 100), std::tuple(100, 100, 100),
                 std::tuple(40, 40, 1000)});

// Execution policy used to exercise the `policy`-first overloads.
struct test_policy {};

// Dense row-major copy of a sparse (or dense) matrix; duplicates are summed.
template <typename M>
auto to_dense(M&& m) {
  using T = std::remove_cvref_t<spblas::tensor_scalar_t<M>>;
  auto shape = spblas::__backend::shape(m);
  std::size_t ncols = shape[1];
  std::vector<T> d(shape[0] * ncols, T(0));
  spblas::__backend::for_each(m, [&](auto&& e) {
    auto&& [idx, v] = e;
    auto&& [i, j] = idx;
    d[i * ncols + j] += v;
  });
  return d;
}

// Number of distinct (i, j) positions stored in a sparse matrix.
template <typename M>
std::size_t count_positions(M&& m) {
  auto shape = spblas::__backend::shape(m);
  std::size_t ncols = shape[1];
  std::vector<bool> seen(shape[0] * ncols, false);
  std::size_t count = 0;
  spblas::__backend::for_each(m, [&](auto&& e) {
    auto&& [idx, v] = e;
    auto&& [i, j] = idx;
    if (!seen[i * ncols + j]) {
      seen[i * ncols + j] = true;
      count++;
    }
  });
  return count;
}

// Whether the indices within every compressed slice (CSR row, CSC column)
// are strictly increasing, i.e. sorted without duplicates.
template <typename P, typename N>
bool sorted_slices(const P& pointers, const N& indices) {
  for (std::size_t o = 0; o + 1 < pointers.size(); o++) {
    for (auto p = pointers[o] + 1; p < pointers[o + 1]; p++) {
      if (!(indices[p - 1] < indices[p])) {
        return false;
      }
    }
  }
  return true;
}

// Sorts a sparse output (whose order within rows / columns is
// backend-defined) with sort_matrix, and checks it is sorted. Call before
// validating the output. Also checks the output's sort_state() claim, which
// a backend sets in the fill stages: `sorted` must hold before sorting.
template <typename C>
void sort_output(C&& c) {
  auto pointers = spblas::__detail::outer_pointers(c);
  auto indices = spblas::__detail::inner_indices(c);
  if (c.sort_state() == spblas::sort_state::sorted) {
    EXPECT_TRUE(sorted_slices(pointers, indices))
        << "the output claims to be sorted, but is not";
  } else if (c.sort_state() == spblas::sort_state::unsorted) {
    EXPECT_FALSE(sorted_slices(pointers, indices))
        << "the output claims to be unsorted, but is sorted";
  }

  spblas::operation_state state;
  spblas::sort_matrix(state, c);
  EXPECT_TRUE(sorted_slices(pointers, indices));
  EXPECT_EQ(c.sort_state(), spblas::sort_state::sorted);
}

// Shuffles the indices (and values) within every compressed slice.
template <typename P, typename N, typename V>
void shuffle_slices(const P& pointers, N& indices, V& values,
                    unsigned seed = 7) {
  std::mt19937 g(seed);
  std::vector<std::size_t> order;
  for (std::size_t o = 0; o + 1 < pointers.size(); o++) {
    std::size_t first = pointers[o], last = pointers[o + 1];
    order.resize(last - first);
    std::iota(order.begin(), order.end(), first);
    std::shuffle(order.begin(), order.end(), g);
    std::vector<typename N::value_type> i(order.size());
    std::vector<typename V::value_type> v(order.size());
    for (std::size_t q = 0; q < order.size(); q++) {
      i[q] = indices[order[q]];
      v[q] = values[order[q]];
    }
    std::copy(i.begin(), i.end(), indices.begin() + first);
    std::copy(v.begin(), v.end(), values.begin() + first);
  }
}

// Values with some variety, so wrong index mappings change the result.
template <typename T>
std::vector<T> make_values(std::size_t n, int offset = 0) {
  std::vector<T> v(n);
  for (std::size_t i = 0; i < n; i++) {
    v[i] = T((i + offset) % 7 + 1) / T(4);
  }
  return v;
}

} // namespace util
