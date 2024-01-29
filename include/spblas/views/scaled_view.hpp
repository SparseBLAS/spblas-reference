#pragma once

#include <spblas/backend/concepts.hpp>
#include <spblas/backend/cpos.hpp>
#include <spblas/concepts.hpp>

namespace spblas {

// NOTE: std::linalg refers to `ScalingFactor` type with name `alpha`.
// Here, I write `S` instead of `ScalingFactor`.

// Scale a tensor of type `T` by a scaling factor of type `S`.
template <typename S, typename T>
class scaled_view;

template <typename S, vector V>
  requires(__detail::view<V>)
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

} // namespace spblas
