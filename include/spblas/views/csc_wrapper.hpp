#pragma once

#include <span>
#include <spblas/detail/detail.hpp>
#include <spblas/views/view_base.hpp>

namespace spblas {

// Non-owning wrapper around user-owned CSC arrays.
template <typename T, std::integral I = index_t, std::integral O = I>
class csc_wrapper : public view_base {
public:
  using scalar_type = T;
  using scalar_reference = T&;
  using index_type = I;
  using offset_type = O;

  csc_wrapper(T* values, O* colptr, I* rowind, index<I> shape, O nnz)
      : values_(values, nnz), colptr_(colptr, shape[1] + 1),
        rowind_(rowind, nnz), shape_(shape), nnz_(nnz) {
    if (colptr_.data() == nullptr) {
      colptr_ = std::span<O>((O*) nullptr, (O*) nullptr);
    }
  }

  template <__ranges::contiguous_range V, __ranges::contiguous_range R,
            __ranges::contiguous_range C>
  csc_wrapper(V&& values, R&& colptr, C&& rowind, index<I> shape, O nnz)
      : values_(__ranges::data(values), __ranges::size(values)),
        colptr_(__ranges::data(colptr), __ranges::size(colptr)),
        rowind_(__ranges::data(rowind), __ranges::size(rowind)), shape_(shape),
        nnz_(nnz) {}

  // Shape-only wrapper for an output matrix whose arrays are attached later
  // with `update()`, e.g. after reading `operation_state::get_result_nnz()`.
  // (No index<I> overload: it would make `csc_wrapper C({m, n})` ambiguous.)
  csc_wrapper(I nrows, I ncols) : shape_(nrows, ncols), nnz_(0) {}

  // Attach arrays; nnz becomes `rowind.size()`, so the arrays must be sized
  // exactly to the number of stored entries. `values` may still be empty
  // while only the sparsity pattern is being filled.
  void update(std::span<T> values, std::span<O> colptr, std::span<I> rowind) {
    values_ = values;
    colptr_ = colptr;
    rowind_ = rowind;
    nnz_ = static_cast<O>(rowind.size());
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
