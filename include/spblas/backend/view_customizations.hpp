#pragma once

#include <spblas/backend/cpos.hpp>

namespace spblas {

// Customization point implementations for csr_view.

template <matrix M>
requires(__detail::is_csr_view_v<M>)
auto tag_invoke(__backend::shape_fn_, M&& m) {
  return m.shape();
}

template <matrix M>
requires(__detail::is_csr_view_v<M>)
auto tag_invoke(__backend::values_fn_, M&& m) {
  return m.values();
}

namespace {

template <matrix M>
requires(__detail::is_csr_view_v<M>)
auto row(M&& m, std::size_t row_index) {
  using O = typename std::remove_cvref_t<M>::offset_type;
  O first = m.rowptr()[row_index];
  O last = m.rowptr()[row_index + 1];

  __ranges::subrange column_indices(__ranges::next(m.colind().data(), first),
                                    __ranges::next(m.colind().data(), last));
  __ranges::subrange row_values(__ranges::next(m.values().data(), first),
                                __ranges::next(m.values().data(), last));

  return __ranges::views::zip(column_indices, row_values);
}

}

template <matrix M>
requires(__detail::is_csr_view_v<M>)
auto tag_invoke(__backend::rows_fn_, M&& m) {
  using I = typename std::remove_cvref_t<M>::index_type;
  auto row_indices = __ranges::views::iota(I(0), I(m.shape()[0]));

  auto row_values =
       row_indices | __ranges::views::transform(
                          [=](auto row_index) { return row(m, row_index); });

  return __ranges::views::zip(row_indices, row_values);
}

// Customization point implementations for contiguous_range

namespace __backend {

template <vector V>
requires(__ranges::contiguous_range<V>)
auto tag_invoke(__backend::shape_fn_, V&& v) {
  return __ranges::size(v);
}

template <vector V>
requires(__ranges::contiguous_range<V>)
auto tag_invoke(__backend::values_fn_, V&& v) {
  return __ranges::views::all(std::forward<V>(v));
}

}

} // namespace spblas