#pragma once

#include <complex>
#include <type_traits>

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

} // namespace spblas
