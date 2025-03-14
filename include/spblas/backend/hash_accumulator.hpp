#pragma once

#include <functional>
#include <span>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <spblas/detail/ranges.hpp>

namespace spblas {

namespace __backend {

template <typename T, std::integral I>
class hash_accumulator {
public:
  hash_accumulator(I count) {}

  T& operator[](I pos) {
    return hash_[pos];
  }

  bool contains(I pos) {
    return hash_.contains(pos);
  }

  void clear() {
    hash_.clear();
  }

  I size() const {
    return hash_.size();
  }

  bool empty() {
    return hash_.empty();
  }

  void sort() {}

  auto get() {
    std::vector<std::pair<I, T>> values(hash_.begin(), hash_.end());

    std::sort(values.begin(), values.end(), [](auto&& a, auto&& b) {
      return std::get<0>(a) < std::get<0>(b);
    });

    return values;
  }

private:
  std::unordered_map<I, T> hash_;
};

template <std::integral T>
class hash_set {
public:
  hash_set(T count) {}

  void insert(T key) {
    set_.insert(key);
  }

  bool contains(T key) {
    return set_.contains(key);
  }

  void clear() {
    set_.clear();
  }

  T size() const {
    return set_.size();
  }

  bool empty() {
    return set_.empty();
  }

  auto get() const {
    return __ranges::views::all(set_);
  }

private:
  std::unordered_set<T> set_;
};

} // namespace __backend

} // namespace spblas
