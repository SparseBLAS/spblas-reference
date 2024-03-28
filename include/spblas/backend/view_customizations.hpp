#pragma once

#include <spblas/backend/cpos.hpp>
#include <spblas/detail/ranges.hpp>

namespace spblas {

// Customization point implementations for csr_view.

template <typename M>
  requires(__detail::is_csr_view_v<M>)
auto tag_invoke(__backend::size_fn_, M&& m) {
  return m.size();
}

template <typename M>
  requires(__detail::is_csr_view_v<M>)
auto tag_invoke(__backend::shape_fn_, M&& m) {
  return m.shape();
}

template <typename M>
  requires(__detail::is_csr_view_v<M>)
auto tag_invoke(__backend::values_fn_, M&& m) {
  return m.values();
}

namespace {

template <typename M>
  requires(__detail::is_csr_view_v<M>)
auto row(M&& m, typename std::remove_cvref_t<M>::index_type row_index) {
  using O = typename std::remove_cvref_t<M>::offset_type;
  O first = m.rowptr()[row_index];
  O last = m.rowptr()[row_index + 1];

  using column_iter_t = decltype(m.colind().data());
  using value_iter_t = decltype(m.values().data());

  __ranges::subrange<column_iter_t> column_indices(
      __ranges::next(m.colind().data(), first),
      __ranges::next(m.colind().data(), last));
  __ranges::subrange<value_iter_t> row_values(
      __ranges::next(m.values().data(), first),
      __ranges::next(m.values().data(), last));

  return __ranges::views::zip(column_indices, row_values);
}

} // namespace

template <typename M>
  requires(__detail::is_csr_view_v<M>)
auto tag_invoke(__backend::rows_fn_, M&& m) {
  using I = typename std::remove_cvref_t<M>::index_type;
  auto row_indices = __ranges::views::iota(I(0), I(m.shape()[0]));

  auto row_values =
      row_indices | __ranges::views::transform(
                        [=](auto row_index) { return row(m, row_index); });

  return __ranges::views::zip(row_indices, row_values);
}

template <typename M>
  requires(__detail::is_csr_view_v<M>)
auto tag_invoke(__backend::lookup_row_fn_, M&& m,
                typename std::remove_cvref_t<M>::index_type row_index) {
  using I = typename std::remove_cvref_t<M>::index_type;
  return row(m, row_index);
}

// Customization point implementations for vectors

template <__ranges::random_access_range V>
struct tensor_traits<V> {
  using scalar_type = __ranges::range_value_t<V>;
  using scalar_reference = __ranges::range_reference_t<V>;
  using index_type = __ranges::range_size_t<V>;
  using offset_type = __ranges::range_size_t<V>;
};

namespace __backend {

template <__ranges::random_access_range V>
auto tag_invoke(__backend::size_fn_, V&& v) {
  return __ranges::size(v);
}

template <__ranges::random_access_range V>
auto tag_invoke(__backend::shape_fn_, V&& v) {
  return __ranges::size(v);
}

template <__ranges::random_access_range V>
auto tag_invoke(__backend::values_fn_, V&& v) {
  return __ranges::views::all(std::forward<V>(v));
}

template <__ranges::random_access_range V>
__ranges::range_reference_t<V> tag_invoke(__backend::lookup_fn_, V&& v,
                                          __ranges::range_size_t<V> i) {
  return *(__ranges::begin(v) + i);
}

} // namespace __backend

// Customization point implementations for mdspan

template <typename M>
  requires(__detail::is_matrix_instantiation_of_mdspan_v<M>)
struct tensor_traits<M> {
  using scalar_type = typename std::remove_cvref_t<M>::value_type;
  using scalar_reference = typename std::remove_cvref_t<M>::reference;
  using index_type = typename std::remove_cvref_t<M>::index_type;
  using offset_type = typename std::remove_cvref_t<M>::size_type;
};

namespace __backend {

template <typename M>
  requires(__detail::is_matrix_instantiation_of_mdspan_v<M>)
auto tag_invoke(__backend::size_fn_, M&& m) {
  return m.extent(0) * m.extent(1);
}

template <typename M>
  requires(__detail::is_matrix_instantiation_of_mdspan_v<M>)
auto tag_invoke(__backend::shape_fn_, M&& m) {
  using index_type = decltype(m.extent(0));
  return index<index_type>(m.extent(0), m.extent(1));
}

template <typename T, typename Extents, typename LayoutPolicy,
          typename AccessorPolicy>
  requires(std::is_same_v<AccessorPolicy, __mdspan::default_accessor<T>> &&
           (std::is_same_v<LayoutPolicy, __mdspan::layout_right> ||
            std::is_same_v<LayoutPolicy, __mdspan::layout_left>) )
auto tag_invoke(__backend::values_fn_,
                __mdspan::mdspan<T, Extents, LayoutPolicy, AccessorPolicy> m) {
  auto size = shape(m)[0] * shape(m)[1];
  return std::span(m.data_handle(), size);
}

// Generic implementation of rows customization point
// for any mdspan.
template <typename T, typename Extents, typename LayoutPolicy,
          typename AccessorPolicy>
auto tag_invoke(__backend::rows_fn_,
                __mdspan::mdspan<T, Extents, LayoutPolicy, AccessorPolicy> m) {
  using index_type = tensor_index_t<decltype(m)>;
  using reference =
      __mdspan::mdspan<T, Extents, LayoutPolicy, AccessorPolicy>::reference;

  auto row_indices = __ranges::views::iota(index_type(0), m.extent(0));

  auto rows =
      row_indices | __ranges::views::transform([=](auto row_index) {
        auto column_indices = __ranges::views::iota(index_type(0), m.extent(1));
        auto values =
            column_indices |
            __ranges::views::transform([=](auto column_index) -> reference {
              return __backend::lookup(m, row_index, column_index);
            });
        return __ranges::views::zip(column_indices, values);
      });
  return __ranges::views::zip(row_indices, rows);
}

template <typename M>
  requires(__detail::is_matrix_instantiation_of_mdspan_v<M>)
tensor_scalar_reference_t<M> tag_invoke(__backend::lookup_fn_, M&& m,
                                        tensor_index_t<M> i,
                                        tensor_index_t<M> j) {
#if defined(__cpp_multidimensional_subscript)
  return m[i, j];
#else
  return m(i, j);
#endif
}

} // namespace __backend

} // namespace spblas
