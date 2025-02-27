#pragma once

#include <algorithm>
#include <memory>
#include <numeric>
#include <random>
#include <set>
#include <vector>

#include <spblas/detail/ranges.hpp>

namespace spblas {

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
  std::uniform_real_distribution d_val(0.0, 100.0);
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
    values.push_back(d_val(g));
  }

  __ranges::sort(__ranges::views::zip(rowind, colind, values));

  return std::tuple(values, rowind, colind, spblas::index<I>(m, n), I(nnz));
}

template <typename T = float, typename I = index_t, typename O = I>
auto generate_csr(std::size_t m, std::size_t n, std::size_t nnz,
                  std::size_t seed = 0) {
  auto&& [values, rowind, colind, shape, _] =
      generate_coo<T, I>(m, n, nnz, seed);

  std::vector<O> rowptr(m + 1);
  for (auto row : rowind) {
    rowptr[row]++;
  }
  std::exclusive_scan(rowptr.begin(), rowptr.end(), rowptr.begin(), O{});

  return std::tuple(values, rowptr, colind, shape, I(nnz));
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
  std::uniform_real_distribution d(0.0, 100.0);

  for (std::size_t i = 0; i < m * n; i++) {
    v.push_back(d(g));
  }

  return std::tuple(v, spblas::index(m, n));
}

template <typename T = float>
auto generate_gaussian(std::size_t m, std::size_t n, std::size_t seed = 0) {
  std::vector<T> v;
  v.reserve(m * n);

  std::mt19937 g(seed);
  std::normal_distribution d{0.0, 1.0};

  for (std::size_t i = 0; i < m * n; i++) {
    v.push_back(d(g));
  }

  return std::tuple(v, spblas::index(m, n));
}

} // namespace spblas
