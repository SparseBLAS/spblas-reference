#pragma once

#include <any>
#include <cassert>
#include <concepts>
#include <limits>
#include <tuple>

#include <spblas/detail/tuple_concept.hpp>
#include <spblas/detail/types.hpp>

namespace spblas {

template <std::integral T = spblas::index_t>
class index {
public:
  using index_type = T;

  using first_type = T;
  using second_type = T;

  constexpr index_type operator[](index_type dim) const noexcept {
    if (dim == 0) {
      return first;
    } else {
      return second;
    }
  }

  constexpr index(index_type first, index_type second)
      : first(first), second(second) {}

  template <typename Tuple>
    requires(!std::is_same_v<Tuple, index> && __detail::tuple_like<Tuple, T, T>)
  constexpr index(Tuple tuple) : first(get<0>(tuple)), second(get<1>(tuple)) {}

  template <std::integral U>
  constexpr index(std::initializer_list<U> tuple) {
    assert(tuple.size() == 2);
    first = *tuple.begin();
    second = *(tuple.begin() + 1);
  }

  constexpr bool operator==(const index&) const noexcept = default;

  index() = default;
  ~index() = default;
  index(const index&) = default;
  index& operator=(const index&) = default;
  index(index&&) = default;
  index& operator=(index&&) = default;

  index_type first;
  index_type second;
};

template <std::size_t Index, std::integral I>
inline constexpr I get(spblas::index<I> index)
  requires(Index <= 1)
{
  if constexpr (Index == 0) {
    return index.first;
  }
  if constexpr (Index == 1) {
    return index.second;
  }
}

} // namespace spblas

namespace std {

template <std::size_t Index, std::integral I>
struct tuple_element<Index, spblas::index<I>>
    : tuple_element<Index, std::tuple<I, I>> {};

template <std::integral I>
struct tuple_size<spblas::index<I>> : integral_constant<std::size_t, 2> {};

template <std::size_t Index, std::integral I>
inline constexpr I get(spblas::index<I> index)
  requires(Index <= 1)
{
  if constexpr (Index == 0) {
    return index.first;
  }
  if constexpr (Index == 1) {
    return index.second;
  }
}

} // namespace std
