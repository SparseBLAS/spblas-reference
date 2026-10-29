#pragma once

#include <functional>
#include <memory_resource>
#include <span>
#include <tuple>
#include <vector>

#include <spblas/detail/ranges.hpp>

namespace spblas {

namespace __backend {

// Dense (sparse accumulator, SPA) row accumulators. Their storage comes from
// `allocator`; algorithms pass their operation_state's.

template <typename T, std::integral I>
class spa_accumulator {
public:
  spa_accumulator(I count, std::pmr::polymorphic_allocator<> allocator = {})
      : data_(count, allocator), set_(count, false, allocator),
        stored_(allocator) {}

  T& operator[](I pos) {
    if (!set_[pos]) {
      stored_.push_back(pos);
      set_[pos] = true;
    }
    return data_[pos];
  }

  bool contains(I pos) {
    return set_[pos];
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
  std::pmr::vector<T> data_;
  std::pmr::vector<bool> set_;
  std::pmr::vector<I> stored_;
};

template <std::integral T>
class spa_set {
public:
  spa_set(T count, std::pmr::polymorphic_allocator<> allocator = {})
      : set_(count, false, allocator), stored_(allocator) {}

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
  std::pmr::vector<bool> set_;
  std::pmr::vector<T> stored_;
};

} // namespace __backend

} // namespace spblas
