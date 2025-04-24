#pragma once

#include <sycl.hpp>

namespace spblas {
namespace mkl {

template <typename T, std::size_t Alignment = 0>
class mkl_allocator {
public:
  using value_type = T;
  using pointer = T*;
  using const_pointer = const T*;
  using reference = T&;
  using const_reference = const T&;
  using size_type = std::size_t;
  using difference_type = std::ptrdiff_t;

  mkl_allocator() noexcept {
    auto* queue = new sycl::queue{sycl::default_selector_v};
    queue_manager_ =
        std::move(std::shared_ptr<sycl::queue>{queue, [](sycl::queue* q) {
                                                 q->wait_and_throw();
                                                 delete q;
                                               }});
  }

  mkl_allocator(sycl::queue* q) noexcept
      : queue_manager_(q, [](sycl::queue* q) {}) {}

  template <typename U>
  mkl_allocator(const mkl_allocator<U, Alignment>& other) noexcept
      : queue_manager_(other.queue_) {}

  mkl_allocator(const mkl_allocator&) = default;
  mkl_allocator& operator=(const mkl_allocator&) = default;
  ~mkl_allocator() = default;

  using is_always_equal = std::false_type;

  pointer allocate(std::size_t size) {
    return sycl::malloc_device<value_type>(size, *(this->queue()));
  }

  void deallocate(pointer ptr, std::size_t n = 0) {
    if (ptr != nullptr) {
      sycl::free(ptr, *(this->queue()));
    }
  }

  bool operator==(const mkl_allocator&) const = default;
  bool operator!=(const mkl_allocator&) const = default;

  template <typename U>
  struct rebind {
    using other = mkl_allocator<U, Alignment>;
  };

  sycl::queue* queue() const noexcept {
    return queue_manager_.get();
  }

private:
  // using shared_ptr to support copy constructor
  std::shared_ptr<sycl::queue> queue_manager_;
};

} // namespace mkl
} // namespace spblas
