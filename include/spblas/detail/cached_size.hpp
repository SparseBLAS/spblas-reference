#pragma once

#include <atomic>
#include <concepts>
#include <limits>

namespace spblas {

namespace __detail {

// A size computed on first use and kept, for views whose size() is not O(1)
// (SparseBLAS paper: views report their size, "through the use of internal
// caching, ... fast after the first time it is computed").
//
// Copies keep a size already computed. Concurrent first calls may both
// compute it; they store the same value. A view may cache its size because it
// depends only on the sparsity pattern, which must not change while it is
// wrapped (a new pattern needs a new wrapper).
template <std::integral O>
class cached_size {
public:
  cached_size() = default;

  cached_size(const cached_size& other) noexcept
      : value_(other.value_.load(std::memory_order_relaxed)) {}

  cached_size& operator=(const cached_size& other) noexcept {
    value_.store(other.value_.load(std::memory_order_relaxed),
                 std::memory_order_relaxed);
    return *this;
  }

  template <typename Compute>
  O get(Compute&& compute) const {
    O value = value_.load(std::memory_order_relaxed);
    if (value == unknown) {
      value = compute();
      value_.store(value, std::memory_order_relaxed);
    }
    return value;
  }

private:
  // A size of exactly max() is recomputed on every call.
  static constexpr O unknown = std::numeric_limits<O>::max();
  mutable std::atomic<O> value_{unknown};
};

} // namespace __detail

} // namespace spblas
