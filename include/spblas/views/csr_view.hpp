#pragma once

#include <spblas/detail/detail.hpp>
#include <span>

namespace spblas {

template <typename T,
          std::integral I = index_t,
          std::integral O = index_t>
class csr_view {
public:
  using scalar_type = T;
  using index_type = I;
  using offset_type = O;

  csr_view(I m, I n, O nnz, T* values, I* rowptr, I* colind)
    : values_(values, nnz), rowptr_(rowptr, m+1), colind_(colind, nnz),
      m_(m), n_(n), nnz_(nnz) {}

  template <__ranges::contiguous_range V,
            __ranges::contiguous_range R,
            __ranges::contiguous_range C>
  csr_view(I m, I n, O nnz, V&& values, R&& rowptr, C&& colind)
    : values_(__ranges::data(values), __ranges::size(values)),
      rowptr_(__ranges::data(rowptr), __ranges::size(rowptr)),
      colind_(__ranges::data(colind), __ranges::size(colind)),
      m_(m), n_(n), nnz_(nnz) {}

  std::span<T> values() const noexcept {
    return values_;
  }

  std::span<I> rowptr() const noexcept {
    return rowptr_;
  }

  std::span<O> colind() const noexcept {
    return colind_;
  }

private:
  std::span<T> values_;
  std::span<I> rowptr_;
  std::span<O> colind_;
  I m_, n_;
  O nnz_;
};

} // namespace spblas
