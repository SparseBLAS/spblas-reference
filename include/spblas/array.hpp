#pragma once

#include <spblas/allocator.hpp>

#include <iostream>
#include <memory>

namespace spblas::detail {

// It is a class to handle the data by allocator which has auto clearup process
template <typename ValueType>
class array {
public:
  using value_type = ValueType;
  /**
   * @return the number of allocated elements.
   */
  size_t size() const noexcept {
    return size_;
  }

  /**
   * @return the pointer of allocated elements.
   */
  value_type* get_data() noexcept {
    return data_;
  }

  /**
   * @return the const pointer of allocated elements.
   */
  const value_type* get_const_data() const noexcept {
    return data_;
  }

  /**
   * allocate the memory for `size` value_type elements via alloc.
   */
  array(std::shared_ptr<const allocator> alloc, size_t size)
      : size_(size), alloc_(alloc),
        data_(static_cast<value_type*>(
            alloc_->alloc(size * sizeof(value_type)))) {}

  ~array() {
    alloc_->free(data_);
    data_ = nullptr;
    size_ = 0;
  }

private:
  using data_manager =
      std::unique_ptr<value_type[], std::function<void(value_type[])>>;
  size_t size_;
  std::shared_ptr<const allocator> alloc_;
  value_type* data_;
};

} // namespace spblas::detail
