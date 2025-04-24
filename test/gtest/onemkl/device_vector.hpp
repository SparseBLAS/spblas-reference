#pragma once
#include <iterator>
#include <memory>
#include <spblas/vendor/onemkl_sycl/mkl_allocator.hpp>
#include <sycl.hpp>
#include <vector>

namespace thrust {

template <typename InputIt, typename OutputIt>
  requires(std::contiguous_iterator<InputIt> &&
           std::contiguous_iterator<OutputIt>)
OutputIt copy(InputIt first, InputIt last, OutputIt d_first) {
  sycl::queue queue(sycl::default_selector_v);
  using input_value_type = typename std::iterator_traits<InputIt>::value_type;
  using output_value_type = typename std::iterator_traits<OutputIt>::value_type;
  input_value_type* first_ptr = std::to_address(first);
  output_value_type* d_first_ptr = std::to_address(d_first);
  auto num = std::distance(first, last);
  queue.memcpy(d_first_ptr, first_ptr, num * sizeof(input_value_type))
      .wait_and_throw();
  return d_first + num;
}

// incompleted impl for thrust vector in oneMKL just for test usage
template <typename ValueType>
class device_vector {
public:
  device_vector(std::vector<ValueType> host_vector)
      : alloc_{}, size_(host_vector.size()), ptr_(nullptr) {
    ptr_ = alloc_.allocate(size_);
    thrust::copy(host_vector.begin(), host_vector.end(), ptr_);
  }

  ~device_vector() {
    alloc_.deallocate(ptr_, size_);
    ptr_ = nullptr;
  }

  ValueType* begin() {
    return ptr_;
  }

  ValueType* end() {
    return ptr_ + size_;
  }

  // just to give data().get()
  std::shared_ptr<ValueType> data() {
    return std::shared_ptr<ValueType>(ptr_, [](ValueType* ptr) {});
  }

private:
  spblas::mkl::mkl_allocator<ValueType> alloc_;
  std::size_t size_;
  ValueType* ptr_;
};

} // namespace thrust
