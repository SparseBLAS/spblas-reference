#pragma once

#include <complex>
#include <cstdint>

#include <rocsparse/rocsparse.h>

namespace spblas {

using index_t = std::int64_t;
using offset_t = std::int64_t;

namespace detail {

/**
 * mapping the type to rocsparse_datatype
 */
template <typename T>
struct rocsparse_datatype_traits {};

#define MAP_ROCSPARSE_DATATYPE(_type, _value)                                  \
  template <>                                                                  \
  struct rocsparse_datatype_traits<_type> {                                    \
    constexpr static rocsparse_datatype value = _value;                        \
  }

MAP_ROCSPARSE_DATATYPE(float, rocsparse_datatype_f32_r);
MAP_ROCSPARSE_DATATYPE(double, rocsparse_datatype_f64_r);
MAP_ROCSPARSE_DATATYPE(std::complex<float>, rocsparse_datatype_f32_c);
MAP_ROCSPARSE_DATATYPE(std::complex<double>, rocsparse_datatype_f64_c);

#undef MAP_ROCSPARSE_DATATYPE

/**
 * mapping the type to rocsparse_indextype
 */
template <typename T>
struct rocsparse_indextype_traits {};

#define MAP_ROCSPARSE_INDEXTYPE(_type, _value)                                 \
  template <>                                                                  \
  struct rocsparse_indextype_traits<_type> {                                   \
    constexpr static rocsparse_indextype value = _value;                       \
  }

MAP_ROCSPARSE_INDEXTYPE(std::int32_t, rocsparse_indextype_i32);
MAP_ROCSPARSE_INDEXTYPE(std::int64_t, rocsparse_indextype_i64);

#undef MAP_ROCSPARSE_INDEXTYPE

} // namespace detail

/**
 * This is an alias for the `rocsparse_datatype` equivalent of `T`.
 *
 * @tparam T  a type
 *
 * @returns the actual `rocsparse_datatype`
 */
template <typename T>
constexpr rocsparse_datatype to_rocsparse_datatype() {
  return detail::rocsparse_datatype_traits<T>::value;
}

/**
 * This is an alias for the `rocsparse_indextype` equivalent of `T`.
 *
 * @tparam T  a type
 *
 * @returns the actual `rocsparse_indextype`
 */
template <typename T>
constexpr rocsparse_indextype to_rocsparse_indextype() {
  return detail::rocsparse_indextype_traits<T>::value;
}

} // namespace spblas
