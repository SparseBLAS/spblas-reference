#pragma once

#include <functional>
#include <span>
#include <spblas/detail/ranges.hpp>
#include <tuple>
#include <vector>

#include <spblas/detail/ranges.hpp>

namespace spblas {

namespace __backend {

template <typename T, std::integral I>
class spa_accumulator {
public:
  spa_accumulator(I count) : data_(count), set_(count, false) {}

  T& operator[](I pos) {
    if (!set_[pos]) {
      stored_.push_back(pos);
      set_[pos] = true;
    }
    return data_[pos];
  }

  void clear() {
    for (auto&& pos : stored_) {
      set_[pos] = false;
      data_[pos] = 0;
    }
    stored_.clear();
  }

  I size() const {
    return stored_.size();
  }

  bool empty() {
    return size() == 0;
  }

  void sort() {
    std::sort(stored_.begin(), stored_.end());
  }

  auto get() {
    std::span data(data_);
    std::span stored(stored_);

    return stored | __ranges::views::transform([=](auto idx) {
             return std::make_tuple(idx, std::reference_wrapper(data[idx]));
           });
  }

private:
  std::vector<T> data_;
  std::vector<bool> set_;
  std::vector<I> stored_;
};

template <std::integral T>
class spa_set {
public:
  spa_set(T count) : set_(count, false) {}

  void insert(T key) {
    if (!set_[key]) {
      stored_.push_back(key);
      set_[key] = true;
    }
  }

  bool contains(T key) {
    return set_[key];
  }

  void clear() {
    for (auto&& pos : stored_) {
      set_[pos] = false;
    }
    stored_.clear();
  }

  T size() const {
    return stored_.size();
  }

  bool empty() {
    return size() == 0;
  }

  auto get() const {
    return std::span(stored_);
  }

private:
  std::vector<bool> set_;
  std::vector<T> stored_;
};

} // namespace __backend

} // namespace spblas
