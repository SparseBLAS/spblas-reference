#pragma once

#include <complex>

#include <spblas/backend/concepts.hpp>
#include <spblas/backend/cpos.hpp>
#include <spblas/concepts.hpp>
#include <spblas/detail/ranges.hpp>
#include <spblas/views/conjugated_view.hpp>
#include <spblas/views/view_base.hpp>

namespace spblas {

// Conjugate a tensor of type `T`.
template <typename T>
class conjugated_view;

// Conjugated view for random access range
template <vector V>
  requires(__detail::__ranges::view<V> && __ranges::random_access_range<V>)
class conjugated_view<V> : public view_base {
public:
  using scalar_type = decltype(std::conj(std::declval<tensor_scalar_t<V>>()));
  using scalar_reference = scalar_type;
  using index_type = tensor_index_t<V>;
  using offset_type = tensor_offset_t<V>;

  explicit conjugated_view(V vector)
      : vector_(vector), transform_view_(vector, transform_fn_{}) {}

  index_type shape() const noexcept {
    return __backend::shape(base());
  }

  index_type size() const noexcept {
    return __backend::size(base());
  }

  scalar_type operator[](index_type i) const {
    return transform_view_[i];
  }

  auto base() {
    return vector_;
  }

  auto base() const {
    return vector_;
  }

  auto begin() {
    return transform_view_.begin();
  }

  auto begin() const {
    return transform_view_.begin();
  }

  auto end() {
    return transform_view_.end();
  }

  auto end() const {
    return transform_view_.end();
  }

private:
  struct transform_fn_ {
    auto operator()(auto x) {
      return std::conj(x);
    }
  };

  __ranges::transform_view<V, transform_fn_> transform_view_;

private:
  V vector_;
};

template <__ranges::random_access_range R>
conjugated_view(R&& r) -> conjugated_view<__ranges::views::all_t<R>>;

// Conjugated view for matrices
template <matrix M>
  requires(view<M>)
class conjugated_view<M> : public view_base {
public:
  using scalar_type = decltype(std::conj(std::declval<tensor_scalar_t<M>>()));
  using scalar_reference = scalar_type;
  using index_type = tensor_index_t<M>;
  using offset_type = tensor_offset_t<M>;

  explicit conjugated_view(M matrix) : matrix_(matrix) {}

  auto shape() const noexcept {
    return __backend::shape(base());
  }

  index_type size() const noexcept {
    return __backend::size(base());
  }

  auto base() {
    return matrix_;
  }

  auto base() const {
    return matrix_;
  }

private:
  friend auto tag_invoke(__backend::size_fn_, conjugated_view matrix) {
    return __backend::size(matrix.base());
  }

  friend auto tag_invoke(__backend::shape_fn_, conjugated_view matrix) {
    return __backend::shape(matrix.base());
  }

  friend auto tag_invoke(__backend::lookup_fn_, conjugated_view matrix,
                         index_type i, index_type j)
    requires(__backend::lookupable<M>)
  {
    return std::conj(__backend::lookup(matrix.base(), i, j));
  }

  friend auto tag_invoke(__backend::rows_fn_, conjugated_view matrix)
    requires(__backend::row_iterable<M>)
  {
    auto unscaled_rows = __backend::rows(matrix.base());

    return unscaled_rows |
           __ranges::views::transform([](auto&& row_tuple) {
             auto&& [column_index, row] = row_tuple;

             auto conjugated_row =
                 row | __ranges::views::transform([](auto&& element_tuple) {
                   auto&& [column_index, value] = element_tuple;
                   return std::pair(column_index, std::conj(value));
                 });

             return std::pair(column_index, conjugated_row);
           });
  }

  friend auto tag_invoke(__backend::lookup_row_fn_, conjugated_view matrix,
                         index_type row_index)
    requires(__backend::row_lookupable<M>)
  {
    auto unscaled_row = __backend::lookup_row(matrix.base(), row_index);

    return unscaled_row | __ranges::views::transform([](auto&& element_tuple) {
             auto&& [column_index, value] = element_tuple;
             return std::pair(column_index, std::conj(value));
           });
  }

  friend auto tag_invoke(__backend::columns_fn_, conjugated_view matrix)
    requires(__backend::column_iterable<M>)
  {
    auto unscaled_columns = __backend::columns(matrix.base());

    return unscaled_columns |
           __ranges::views::transform([](auto&& column_tuple) {
             auto&& [row_index, column] = column_tuple;

             auto conjugated_column =
                 column | __ranges::views::transform([](auto&& element_tuple) {
                   auto&& [row_index, value] = element_tuple;
                   return std::pair(row_index, std::conj(value));
                 });

             return std::pair(row_index, conjugated_column);
           });
  }

  friend auto tag_invoke(__backend::lookup_column_fn_, conjugated_view matrix,
                         index_type column_index)
    requires(__backend::column_lookupable<M>)
  {
    auto unscaled_column =
        __backend::lookup_column(matrix.base(), column_index);

    return unscaled_column | __ranges::views::transform([](auto&& element_tuple) {
             auto&& [row_index, value] = element_tuple;
             return std::pair(row_index, std::conj(value));
           });
  }

private:
  M matrix_;
};

template <matrix M>
  requires(view<M>)
conjugated_view(M m) -> conjugated_view<M>;

} // namespace spblas
