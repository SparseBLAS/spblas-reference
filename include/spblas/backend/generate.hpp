#pragma once

#include <algorithm>
#include <memory>
#include <numeric>
#include <random>
#include <set>
#include <tuple>
#include <type_traits>
#include <vector>

#include <spblas/detail/ranges.hpp>
#include <spblas/detail/type_traits.hpp>

namespace spblas {

namespace __detail {

template <typename T, typename G>
T random_uniform(G& g, T lo, T hi) {
  if constexpr (is_std_complex_v<T>) {
    using value_type = typename std::remove_cvref_t<T>::value_type;
    std::uniform_real_distribution<value_type> d(lo.real(), hi.real());
    return T(d(g), d(g));
  } else if constexpr (std::is_integral_v<T>) {
    std::uniform_int_distribution<T> d(lo, hi);
    return d(g);
  } else {
    std::uniform_real_distribution<T> d(lo, hi);
    return d(g);
  }
}

template <typename T, typename G>
T random_gaussian(G& g, T mean, T stddev) {
  if constexpr (is_std_complex_v<T>) {
    using value_type = typename std::remove_cvref_t<T>::value_type;
    std::normal_distribution<value_type> d(mean.real(), stddev.real());
    return T(d(g), d(g));
  } else {
    std::normal_distribution<T> d(mean, stddev);
    return d(g);
  }
}

} // namespace __detail

template <typename T = float, typename I = index_t>
auto generate_coo(std::size_t m, std::size_t n, std::size_t nnz,
                  std::size_t seed = 0) {
  std::vector<T> values;
  std::vector<I> rowind;
  std::vector<I> colind;

  values.reserve(nnz);
  rowind.reserve(nnz);
  colind.reserve(nnz);

  std::mt19937 g(seed);
  std::uniform_int_distribution<I> d_row(0, m - 1);
  std::uniform_int_distribution<I> d_col(0, n - 1);
  std::set<std::pair<I, I>> entries;

  for (std::size_t i = 0; i < nnz; i++) {
    auto row = d_row(g);
    auto col = d_col(g);
    while (entries.contains({row, col})) {
      row = d_row(g);
      col = d_col(g);
    }
    entries.emplace(row, col);
    rowind.push_back(row);
    colind.push_back(col);
    values.push_back(__detail::random_uniform<T>(g, T{0}, T{100}));
  }

  __ranges::sort(__ranges::views::zip(rowind, colind, values),
                 [](const auto& left, const auto& right) {
                   if (std::get<0>(left) < std::get<0>(right)) {
                     return true;
                   }
                   if (std::get<0>(right) < std::get<0>(left)) {
                     return false;
                   }
                   return std::get<1>(left) < std::get<1>(right);
                 });

  return std::tuple(values, rowind, colind, spblas::index<I>(m, n), I(nnz));
}

template <typename T = float, typename I = index_t, typename O = I>
auto generate_csr_sorted(std::size_t m, std::size_t n, std::size_t nnz,
                         std::size_t seed = 0) {
  auto&& [values, rowind, colind, shape, nnz_] =
      generate_coo<T, I>(m, n, nnz, seed);

  std::vector<O> rowptr(m + 1);
  for (auto row : rowind) {
    rowptr[row]++;
  }
  std::exclusive_scan(rowptr.begin(), rowptr.end(), rowptr.begin(), O{});

  return std::tuple(values, rowptr, colind, shape, I(nnz));
}

template <typename T = float, typename I = index_t, typename O = I>
auto generate_csr(std::size_t m, std::size_t n, std::size_t nnz,
                  std::size_t seed = 0) {
  auto&& [values, rowptr, colind, shape, nnz_] =
      generate_csr_sorted<T, I, O>(m, n, nnz, seed);

  for (std::size_t row = 0; row < m; row++) {
    const auto row_begin = rowptr[row];
    const auto row_end = rowptr[row + 1];
    std::shuffle(colind.begin() + row_begin, colind.begin() + row_end,
                 std::mt19937(seed));
  }

  return std::tuple(values, rowptr, colind, shape, I(nnz));
}

template <typename T = float, typename I = index_t, typename O = I>
auto generate_csc_sorted(std::size_t m, std::size_t n, std::size_t nnz,
                         std::size_t seed = 0) {
  auto&& [values, colptr, rowind, shape_, nnz_] =
      generate_csr_sorted<T, I, O>(n, m, nnz, seed);
  return std::tuple(std::move(values), std::move(colptr), std::move(rowind),
                    spblas::index<I>(m, n), I(nnz));
}

template <typename T = float, typename I = index_t, typename O = I>
auto generate_csc(std::size_t m, std::size_t n, std::size_t nnz,
                  std::size_t seed = 0) {
  auto&& [values, colptr, rowind, shape_, nnz_] =
      generate_csr<T, I, O>(n, m, nnz, seed);
  return std::tuple(std::move(values), std::move(colptr), std::move(rowind),
                    spblas::index<I>(m, n), I(nnz));
}

template <typename T = float, typename I = index_t, typename O = I>
auto generate_dcsr(std::size_t m, std::size_t n, std::size_t nnz,
                   std::size_t seed = 0) {
  auto&& [values, rowptr_, colind, shape, _] =
      generate_csr<T, I, O>(m, n, nnz, seed);

  std::size_t num_rows = 0;

  for (std::size_t i = 0; i < m; i++) {
    if (rowptr_[i + 1] - rowptr_[i] > 0) {
      num_rows++;
    }
  }

  std::vector<O> rowptr(num_rows + 1);
  std::vector<I> rowind(num_rows);

  O rp_idx = 0;
  for (std::size_t i = 0; i < m; i++) {
    if (rowptr_[i + 1] - rowptr_[i] > 0) {
      rowptr[rp_idx] = rowptr_[i];
      rowind[rp_idx] = i;
      rp_idx++;
    }
  }
  rowptr[num_rows] = nnz;

  return std::tuple(values, rowind, rowptr, colind, shape, num_rows, nnz);
}

template <typename T = float>
auto generate_dense(std::size_t m, std::size_t n, std::size_t seed = 0) {
  std::vector<T> v;
  v.reserve(m * n);

  std::mt19937 g(seed);

  for (std::size_t i = 0; i < m * n; i++) {
    v.push_back(__detail::random_uniform<T>(g, T{0}, T{100}));
  }

  return std::tuple(v, spblas::index(m, n));
}

template <typename T = float>
auto generate_gaussian(std::size_t m, std::size_t n, std::size_t seed = 0) {
  std::vector<T> v;
  v.reserve(m * n);

  std::mt19937 g(seed);

  for (std::size_t i = 0; i < m * n; i++) {
    v.push_back(__detail::random_gaussian<T>(g, T{0}, T{1}));
  }

  return std::tuple(v, spblas::index(m, n));
}

} // namespace spblas
