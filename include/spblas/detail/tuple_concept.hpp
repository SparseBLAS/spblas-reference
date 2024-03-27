#pragma once

#include <any>
#include <concepts>
#include <iterator>

namespace spblas {

namespace __detail {

template <typename T, std::size_t I, typename U = std::any>
concept tuple_element_gettable = requires(T tuple) {
  { get<I>(tuple) } -> std::convertible_to<U>;
};

template <typename T, typename... Args>
concept tuple_like =
    requires {
      typename std::tuple_size<std::remove_cvref_t<T>>::type;
      requires std::same_as<
          std::remove_cvref_t<
              decltype(std::tuple_size_v<std::remove_cvref_t<T>>)>,
          std::size_t>;
    } && sizeof...(Args) == std::tuple_size_v<std::remove_cvref_t<T>> &&
    []<std::size_t... I>(std::index_sequence<I...>) {
      return (tuple_element_gettable<T, I, Args> && ...);
    }(std::make_index_sequence<std::tuple_size_v<std::remove_cvref_t<T>>>());

} // namespace __detail
} // namespace spblas
