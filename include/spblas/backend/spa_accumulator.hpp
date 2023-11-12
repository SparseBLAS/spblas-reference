#pragma once

#include <span>
#include <vector>

namespace spblas {

namespace __backend {

template <typename T>
class spa_accumulator {
public:
  spa_accumulator(std::size_t count) : data_(count), set_(count, false) {}

  T& operator[](std::size_t pos) {
    if (!set_[pos]) {
      stored_.push_back(pos);
      set_[pos] = true;
    }
    return data_[pos];
  }

  void clear() {
    for (auto&& pos : stored_) {
      set_[pos] = false;
    }
    stored_.clear();
  }

  std::size_t size() const { return data_.size(); }

  void sort() { std::sort(stored_.begin(), stored_.end()); }

  auto get() const {
    std::span data(data_);
    std::span stored(stored_);

    return stored | __ranges::views::transform([=](auto idx) {
             return std::tuple{idx, data[idx]};
           });
  }

private:
  std::vector<T> data_;
  std::vector<bool> set_;
  std::vector<std::size_t> stored_;
};

template <std::integral T>
class spa_set {
public:
  spa_set(std::size_t count) : set_(count, false) {}

  void insert(T key) {
    if (!set_[key]) {
      stored_.push_back(key);
      set_[key] = true;
    }
  }

  bool contains(T key) { return set_[key]; }

  void clear() {
    for (auto&& pos : stored_) {
      set_[pos] = false;
    }
    stored_.clear();
  }

  std::size_t size() const { return stored_.size(); }

private:
  std::vector<bool> set_;
  std::vector<std::size_t> stored_;
};

} // namespace __backend

} // namespace spblas
