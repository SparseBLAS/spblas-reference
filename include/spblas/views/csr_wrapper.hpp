#pragma once

#include <span>
#include <spblas/detail/detail.hpp>
#include <spblas/views/view_base.hpp>

namespace spblas {

template <typename T, typename I, typename O>
class csr_builder;

// Non-owning wrapper around user-owned CSR arrays.
template <typename T, std::integral I = index_t, std::integral O = I>
class csr_wrapper : public view_base {
public:
  using scalar_type = T;
  using scalar_reference = T&;
  using index_type = I;
  using offset_type = O;

  csr_wrapper(T* values, O* rowptr, I* colind, index<I> shape, O nnz)
      : values_(values, nnz), rowptr_(rowptr, shape[0] + 1),
        colind_(colind, nnz), shape_(shape), nnz_(nnz) {
    if (rowptr_.data() == nullptr) {
      rowptr_ = std::span<O>((O*) nullptr, (O*) nullptr);
    }
  }

  template <__ranges::contiguous_range V, __ranges::contiguous_range R,
            __ranges::contiguous_range C>
  csr_wrapper(V&& values, R&& rowptr, C&& colind, index<I> shape, O nnz)
      : values_(__ranges::data(values), __ranges::size(values)),
        rowptr_(__ranges::data(rowptr), __ranges::size(rowptr)),
        colind_(__ranges::data(colind), __ranges::size(colind)), shape_(shape),
        nnz_(nnz) {}

  // Shape-only wrapper for an output matrix whose arrays are attached later
  // with `update()`, e.g. after reading `operation_state::get_result_nnz()`.
  // (No index<I> overload: it would make `csr_wrapper C({m, n})` ambiguous.)
  csr_wrapper(I nrows, I ncols) : shape_(nrows, ncols), nnz_(0) {}

  // Attach arrays; nnz becomes `colind.size()`, so the arrays must be sized
  // exactly to the number of stored entries. `values` may still be empty
  // while only the sparsity pattern is being filled.
  void update(std::span<T> values, std::span<O> rowptr, std::span<I> colind) {
    values_ = values;
    rowptr_ = rowptr;
    colind_ = colind;
    nnz_ = static_cast<O>(colind.size());
  }

  void update(std::span<T> values, std::span<O> rowptr, std::span<I> colind,
              index<I> shape, O nnz) {
    values_ = values;
    rowptr_ = rowptr;
    colind_ = colind;
    shape_ = shape;
    nnz_ = nnz;
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

  friend class csr_builder<T, I, O>;

private:
  std::span<T> values_;
  std::span<O> rowptr_;
  std::span<I> colind_;
  index<I> shape_;
  O nnz_;
};

} // namespace spblas
