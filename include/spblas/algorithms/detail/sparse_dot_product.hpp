#pragma once

#include <optional>

#include <spblas/backend/spa_accumulator.hpp>

namespace spblas {

namespace __detail {

template <typename T, typename I, typename A, typename B>
std::optional<T> sparse_dot_product(__backend::spa_accumulator<T, I>& acc,
                                    A&& a, B&& b) {
  acc.clear();

  for (auto&& [i, v] : a) {
    acc[i] = v;
  }

  T sum = 0;
  bool implicit_zero = true;
  for (auto&& [i, v] : b) {
    if (acc.contains(i)) {
      sum += acc[i] * v;
      implicit_zero = false;
    }
  }

  if (implicit_zero) {
    return {};
  } else {
    return sum;
  }
}

template <typename Set, typename A, typename B>
bool sparse_intersection(Set&& set, A&& a, B&& b) {
  set.clear();

  for (auto&& [i, v] : a) {
    set.insert(i);
  }

  for (auto&& [i, v] : b) {
    if (set.contains(i)) {
      return true;
    }
  }

  return false;
}

} // namespace __detail

} // namespace spblas
