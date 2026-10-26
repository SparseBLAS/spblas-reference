#pragma once

#include <span>
#include <spblas/detail/detail.hpp>
#include <spblas/views/properties.hpp>
#include <spblas/views/view_base.hpp>

namespace spblas {

template <typename T, typename I, typename O>
class csr_builder;

// Non-owning wrapper around user-owned CSR arrays, with the properties of
// the data they hold (properties.hpp).
template <typename T, std::integral I = index_t, std::integral O = I>
class csr_wrapper : public view_base {
public:
  using scalar_type = T;
  using scalar_reference = T&;
  using index_type = I;
  using offset_type = O;

  csr_wrapper(T* values, O* rowptr, I* colind, index<I> shape, O nnz,
              property_list properties = {})
      : values_(values, nnz), rowptr_(rowptr, shape[0] + 1),
        colind_(colind, nnz), shape_(shape), nnz_(nnz),
        properties_(properties) {
    if (rowptr_.data() == nullptr) {
      rowptr_ = std::span<O>((O*) nullptr, (O*) nullptr);
    }
  }

  template <__ranges::contiguous_range V, __ranges::contiguous_range R,
            __ranges::contiguous_range C>
  csr_wrapper(V&& values, R&& rowptr, C&& colind, index<I> shape, O nnz,
              property_list properties = {})
      : values_(__ranges::data(values), __ranges::size(values)),
        rowptr_(__ranges::data(rowptr), __ranges::size(rowptr)),
        colind_(__ranges::data(colind), __ranges::size(colind)), shape_(shape),
        nnz_(nnz), properties_(properties) {}

  // Shape-only wrapper for an output matrix whose arrays are attached later
  // with `update()`, e.g. after reading `operation_state::get_result_nnz()`.
  // (No index<I> overload: it would make `csr_wrapper C({m, n})` ambiguous.)
  csr_wrapper(I nrows, I ncols) : shape_(nrows, ncols), nnz_(0) {}

  // Attach arrays; nnz becomes `colind.size()`, so the arrays must be sized
  // exactly to the number of stored entries. `values` may still be empty
  // while only the sparsity pattern is being filled.
  //
  // The properties are kept if the pattern stays the same arrays (same
  // rowptr and colind spans, shape and nnz), e.g. when attaching values to a
  // pattern filled by a _fill_symbolic stage; otherwise they become unknown.
  // The overloads taking a property_list set them instead.
  void update(std::span<T> values, std::span<O> rowptr, std::span<I> colind) {
    update(values, rowptr, colind, shape_, static_cast<O>(colind.size()));
  }

  void update(std::span<T> values, std::span<O> rowptr, std::span<I> colind,
              index<I> shape, O nnz) {
    if (!same_pattern_(rowptr, colind, shape, nnz)) {
      properties_ = {};
    }
    values_ = values;
    rowptr_ = rowptr;
    colind_ = colind;
    shape_ = shape;
    nnz_ = nnz;
  }

  void update(std::span<T> values, std::span<O> rowptr, std::span<I> colind,
              property_list properties) {
    update(values, rowptr, colind);
    properties_ = properties;
  }

  void update(std::span<T> values, std::span<O> rowptr, std::span<I> colind,
              index<I> shape, O nnz, property_list properties) {
    update(values, rowptr, colind, shape, nnz);
    properties_ = properties;
  }

  std::span<T> values() const noexcept {
    return values_;
  }
  std::span<O> rowptr() const noexcept {
    return rowptr_;
  }
  std::span<I> colind() const noexcept {
    return colind_;
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

  // Sortedness of the column indices within every row.
  spblas::sort_state sort_state() const noexcept {
    return properties_.sort_state();
  }

  void set_sort_state(spblas::sort_state s) noexcept {
    properties_.set_sort_state(s);
  }

  friend class csr_builder<T, I, O>;

private:
  bool same_pattern_(std::span<O> rowptr, std::span<I> colind, index<I> shape,
                     O nnz) const noexcept {
    return rowptr.data() == rowptr_.data() && rowptr.size() == rowptr_.size() &&
           colind.data() == colind_.data() && colind.size() == colind_.size() &&
           shape == shape_ && nnz == nnz_;
  }

  std::span<T> values_;
  std::span<O> rowptr_;
  std::span<I> colind_;
  index<I> shape_;
  O nnz_;
  property_list properties_;
};

} // namespace spblas
