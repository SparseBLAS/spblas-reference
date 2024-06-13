#pragma once

#include <spblas/backend/cpos.hpp>
#include <spblas/detail/types.hpp>

namespace spblas {

class allocator {
public:
  virtual void alloc(void** ptrptr, size_t size) const = 0;

  virtual void free(void* ptr) const = 0;
};

namespace __backend {

template <typename T>
concept row_iterable = requires(T& t) { rows(t); };

template <typename T>
concept row_lookupable = requires(T& t) { lookup_row(t, tensor_index_t<T>{}); };

// using the allocate function from member function or static function.
template <typename T>
concept is_allocator = requires(T& t) {
  { t.alloc(std::declval<void**>(), size_t()) };
  { t.free(std::declval<void**>()) };
};

namespace {

template <typename T>
concept lookupable_matrix =
    requires(T& t, tensor_index_t<T> i, tensor_index_t<T> j) {
      { lookup(t, i, j) };
    };

template <typename T>
concept lookupable_vector = requires(T& t, tensor_index_t<T> i) {
  { lookup(t, i) };
};

} // namespace

template <typename T>
concept lookupable = lookupable_matrix<T> || lookupable_vector<T>;

} // namespace __backend

} // namespace spblas
