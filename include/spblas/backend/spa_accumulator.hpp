#pragma once

#include <vector>
#include <span>

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

  std::size_t size() const {
    return data_.size();
  }

  void sort() {
    std::sort(stored_.begin(), stored_.end());
  }

  auto get() const {
    std::span data(data_);
    std::span stored(stored_);

    return stored |
    __ranges::views::transform([=](auto idx) {
                return std::tuple{idx, data[idx]};
            });
  }

private:
  std::vector<T> data_;
  std::vector<bool> set_;
  std::vector<std::size_t> stored_;
};

}

}