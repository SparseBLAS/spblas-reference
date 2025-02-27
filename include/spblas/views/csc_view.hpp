#pragma once

#include <span>
#include <spblas/detail/detail.hpp>
#include <spblas/views/view_base.hpp>

namespace spblas {

template <typename T, std::integral I = index_t, std::integral O = I>
class csc_view : public view_base {
public:
  using scalar_type = T;
  using scalar_reference = T&;
  using index_type = I;
  using offset_type = O;

  csc_view(T* values, O* colptr, I* rowind, index<I> shape, O nnz)
      : values_(values, nnz), colptr_(colptr, shape[1] + 1),
        rowind_(rowind, nnz), shape_(shape), nnz_(nnz) {
    if (colptr_.data() == nullptr) {
      colptr_ = std::span<O>((O*) nullptr, (O*) nullptr);
    }
  }

  template <__ranges::contiguous_range V, __ranges::contiguous_range R,
            __ranges::contiguous_range C>
  csc_view(V&& values, R&& colptr, C&& rowind, index<I> shape, O nnz)
      : values_(__ranges::data(values), __ranges::size(values)),
        colptr_(__ranges::data(colptr), __ranges::size(colptr)),
        rowind_(__ranges::data(rowind), __ranges::size(rowind)), shape_(shape),
        nnz_(nnz) {}

  void update(std::span<T> values, std::span<O> colptr, std::span<I> rowind) {
    values_ = values;
    colptr_ = colptr;
    rowind_ = rowind;
  }

  void update(std::span<T> values, std::span<O> colptr, std::span<I> rowind,
              index<I> shape, O nnz) {
    values_ = values;
    colptr_ = colptr;
    rowind_ = rowind;
    shape_ = shape;
    nnz_ = nnz;
  }

  std::span<T> values() const noexcept {
    return values_;
  }
  std::span<O> colptr() const noexcept {
    return colptr_;
  }
  std::span<I> rowind() const noexcept {
    return rowind_;
  }

  index<I> shape() const noexcept {
    return shape_;
  }

  O size() const noexcept {
    return nnz_;
  }

private:
  std::span<T> values_;
  std::span<O> colptr_;
  std::span<I> rowind_;
  index<I> shape_;
  O nnz_;
};

} // namespace spblas
