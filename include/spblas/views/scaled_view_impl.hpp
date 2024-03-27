#pragma once

#include <spblas/backend/concepts.hpp>
#include <spblas/backend/cpos.hpp>
#include <spblas/concepts.hpp>
#include <spblas/views/scaled_view.hpp>

namespace spblas {

// NOTE: std::linalg refers to `ScalingFactor` type with name `alpha`.
// Here, I write `S` instead of `ScalingFactor`.

// Scale a tensor of type `T` by a scaling factor of type `S`.
template <typename S, typename T>
class scaled_view;

// Scaled view for random access range
template <typename S, vector V>
  requires(__detail::view<V> && __ranges::random_access_range<V>)
class scaled_view<S, V> : public view_base {
public:
  using scalar_type =
      decltype(std::declval<S>() * std::declval<tensor_scalar_t<V>>());
  using scalar_reference = scalar_type;
  using index_type = tensor_index_t<V>;
  using offset_type = tensor_offset_t<V>;

  scaled_view(S alpha, V vector)
      : alpha_(alpha), vector_(vector),
        transform_view_(vector, transform_fn_(alpha)) {}

  index_type shape() const noexcept {
    return __backend::shape(base());
  }

  index_type size() const noexcept {
    return __backend::size(base());
  }

  S alpha() const noexcept {
    return alpha_;
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
    transform_fn_(S alpha) : alpha_(alpha) {}

    auto operator()(auto x) {
      return alpha_ * x;
    }

  private:
    S alpha_;
  };

  __ranges::transform_view<V, transform_fn_> transform_view_;

private:
  S alpha_;
  V vector_;
};

template <typename S, __ranges::random_access_range R>
scaled_view(S alpha, R&& r) -> scaled_view<S, __ranges::views::all_t<R>>;

// Scaled view for matrices
template <typename S, matrix M>
  requires(view<M>)
class scaled_view<S, M> : public view_base {
public:
  using scalar_type =
      decltype(std::declval<S>() * std::declval<tensor_scalar_t<M>>());
  using scalar_reference = scalar_type;
  using index_type = tensor_index_t<M>;
  using offset_type = tensor_offset_t<M>;

  scaled_view(S alpha, M matrix) : alpha_(alpha), matrix_(matrix) {}

  auto shape() const noexcept {
    return __backend::shape(base());
  }

  index_type size() const noexcept {
    return __backend::size(base());
  }

  S alpha() const noexcept {
    return alpha_;
  }

  auto base() {
    return matrix_;
  }

  auto base() const {
    return matrix_;
  }

private:
  friend auto tag_invoke(__backend::size_fn_, scaled_view matrix) {
    return __backend::size(matrix.base());
  }

  friend auto tag_invoke(__backend::shape_fn_, scaled_view matrix) {
    return __backend::shape(matrix.base());
  }

  friend auto tag_invoke(__backend::lookup_fn_, scaled_view matrix,
                         index_type i, index_type j)
    requires(__backend::lookupable<M>)
  {
    return matrix.alpha() * __backend::lookup(matrix.base(), i, j);
  }

  friend auto tag_invoke(__backend::rows_fn_, scaled_view matrix)
    requires(__backend::row_iterable<M>)
  {
    S alpha = matrix.alpha();
    auto unscaled_rows = __backend::rows(matrix.base());

    return unscaled_rows |
           __ranges::views::transform([alpha](auto&& row_tuple) {
             auto&& [column_index, row] = row_tuple;

             auto scaled_row =
                 row |
                 __ranges::views::transform([alpha](auto&& element_tuple) {
                   auto&& [column_index, value] = element_tuple;
                   return std::pair(column_index, alpha * value);
                 });

             return std::pair(column_index, scaled_row);
           });
  }

  friend auto tag_invoke(__backend::lookup_row_fn_, scaled_view matrix,
                         index_type row_index)
    requires(__backend::row_lookupable<M>)
  {
    S alpha = matrix.alpha();
    auto unscaled_row = __backend::lookup_row(matrix.base(), row_index);

    return unscaled_row |
           __ranges::views::transform([alpha](auto&& element_tuple) {
             auto&& [column_index, value] = element_tuple;
             return std::pair(column_index, alpha * value);
           });
  }

private:
  S alpha_;
  M matrix_;
};

template <typename S, matrix M>
  requires(view<M>)
scaled_view(S s, M m) -> scaled_view<S, M>;

} // namespace spblas
