#pragma once

#include <span>
#include <spblas/detail/detail.hpp>
#include <spblas/views/properties.hpp>
#include <spblas/views/view_base.hpp>
#include <stdexcept>

namespace spblas {

template <typename T>
class matrix_opt;

// Non-owning wrapper around user-owned CSC arrays, with the properties of
// the data they hold (properties.hpp).
template <typename T, std::integral I = index_t, std::integral O = I>
class csc_wrapper : public view_base {
public:
  using scalar_type = T;
  using scalar_reference = T&;
  using index_type = I;
  using offset_type = O;

  csc_wrapper(T* values, O* colptr, I* rowind, index<I> shape, O nnz,
              property_list properties = {})
      : values_(values, nnz), colptr_(colptr, shape[1] + 1),
        rowind_(rowind, nnz), shape_(shape), nnz_(nnz),
        properties_(properties) {
    if (colptr_.data() == nullptr) {
      colptr_ = std::span<O>((O*) nullptr, (O*) nullptr);
    }
  }

  template <__ranges::contiguous_range V, __ranges::contiguous_range R,
            __ranges::contiguous_range C>
  csc_wrapper(V&& values, R&& colptr, C&& rowind, index<I> shape, O nnz,
              property_list properties = {})
      : values_(__ranges::data(values), __ranges::size(values)),
        colptr_(__ranges::data(colptr), __ranges::size(colptr)),
        rowind_(__ranges::data(rowind), __ranges::size(rowind)), shape_(shape),
        nnz_(nnz), properties_(properties) {}

  // Shape-only wrapper for an output matrix whose arrays are attached later
  // with `update()`, e.g. after reading `operation_state::get_result_nnz()`.
  // (No index<I> overload: it would make `csc_wrapper C({m, n})` ambiguous.)
  csc_wrapper(I nrows, I ncols) : shape_(nrows, ncols), nnz_(0) {}

  // Attach arrays; nnz becomes `rowind.size()`, so the arrays must be sized
  // exactly to the number of stored entries. `values` may still be empty
  // while only the sparsity pattern is being filled.
  //
  // The properties are kept if the pattern stays the same arrays (same
  // colptr and rowind spans, shape and nnz), e.g. when attaching values to a
  // pattern filled by a _fill_symbolic stage; otherwise they become unknown.
  // The overloads taking a property_list set them instead.
  void update(std::span<T> values, std::span<O> colptr, std::span<I> rowind) {
    update(values, colptr, rowind, shape_, static_cast<O>(rowind.size()));
  }

  void update(std::span<T> values, std::span<O> colptr, std::span<I> rowind,
              index<I> shape, O nnz) {
    if (!same_pattern_(colptr, rowind, shape, nnz)) {
      properties_ = {};
    }
    values_ = values;
    colptr_ = colptr;
    rowind_ = rowind;
    shape_ = shape;
    nnz_ = nnz;
  }

  void update(std::span<T> values, std::span<O> colptr, std::span<I> rowind,
              property_list properties) {
    update(values, colptr, rowind);
    properties_ = properties;
  }

  void update(std::span<T> values, std::span<O> colptr, std::span<I> rowind,
              index<I> shape, O nnz, property_list properties) {
    update(values, colptr, rowind, shape, nnz);
    properties_ = properties;
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

  property_list properties() const noexcept {
    return properties_;
  }

  // Sortedness of the row indices within every column.
  spblas::sort_state sort_state() const noexcept {
    return properties_.sort_state();
  }

  void set_sort_state(spblas::sort_state s) noexcept {
    properties_.set_sort_state(s);
  }

  // Copies the properties from a matrix_opt's copy of this wrapper (e.g.
  // `sorted` after `sort_matrix(state, A_opt)`); the arrays are not touched.
  // The matrix_opt must be over the same colptr and rowind spans, shape and
  // nnz, and the same values span unless either is empty (a pattern only).
  // Otherwise its properties are claims about other data, and this throws
  // std::invalid_argument.
  void sync_properties(const matrix_opt<csc_wrapper>& m) {
    csc_wrapper inner = m.base();
    if (!same_pattern_(inner.colptr_, inner.rowind_, inner.shape_,
                       inner.nnz_) ||
        !(inner.values_.empty() || values_.empty() ||
          (inner.values_.data() == values_.data() &&
           inner.values_.size() == values_.size()))) {
      throw std::invalid_argument(
          "csc_wrapper::sync_properties: the matrix_opt is not over this "
          "wrapper's arrays.");
    }
    properties_ = inner.properties_;
  }

private:
  bool same_pattern_(std::span<O> colptr, std::span<I> rowind, index<I> shape,
                     O nnz) const noexcept {
    return colptr.data() == colptr_.data() && colptr.size() == colptr_.size() &&
           rowind.data() == rowind_.data() && rowind.size() == rowind_.size() &&
           shape == shape_ && nnz == nnz_;
  }

  std::span<T> values_;
  std::span<O> colptr_;
  std::span<I> rowind_;
  index<I> shape_;
  O nnz_;
  property_list properties_;
};

} // namespace spblas
