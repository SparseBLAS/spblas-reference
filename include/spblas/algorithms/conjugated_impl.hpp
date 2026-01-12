#pragma once

#include <complex>
#include <type_traits>
#include <utility>

#include <spblas/concepts.hpp>
#include <spblas/views/conjugated_view.hpp>

namespace spblas {

namespace __detail {

template <typename T>
struct is_std_complex : std::false_type {};

template <typename T>
struct is_std_complex<std::complex<T>> : std::true_type {};

template <typename T>
inline constexpr bool is_std_complex_v =
    is_std_complex<std::remove_cvref_t<T>>::value;

} // namespace __detail

template <vector V>
auto conjugated(V&& v) {
  if constexpr (__detail::is_std_complex_v<tensor_scalar_t<V>>) {
    return conjugated_view(std::forward<V>(v));
  } else {
    return std::forward<V>(v);
  }
}

template <matrix M>
auto conjugated(M&& m) {
  if constexpr (__detail::is_std_complex_v<tensor_scalar_t<M>>) {
    return conjugated_view(std::forward<M>(m));
  } else {
    return std::forward<M>(m);
  }
}

} // namespace spblas
