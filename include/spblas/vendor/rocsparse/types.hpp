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
struct rocsparse_data_type_impl {};

#define MAP_ROCSPARSE_DATA_TYPE(_type, _value)                                 \
  template <>                                                                  \
  struct rocsparse_data_type_impl<_type> {                                     \
    constexpr static rocsparse_datatype value = _value;                        \
  }

MAP_ROCSPARSE_DATA_TYPE(float, rocsparse_datatype_f32_r);
MAP_ROCSPARSE_DATA_TYPE(double, rocsparse_datatype_f64_r);
MAP_ROCSPARSE_DATA_TYPE(std::complex<float>, rocsparse_datatype_f32_c);
MAP_ROCSPARSE_DATA_TYPE(std::complex<double>, rocsparse_datatype_f64_c);

#undef MAP_ROCSPARSE_DATA_TYPE

/**
 * mapping the type to rocsparse_indextype
 */
template <typename T>
struct rocsparse_index_type_impl {};

#define MAP_ROCSPARSE_INDEX_TYPE(_type, _value)                                \
  template <>                                                                  \
  struct rocsparse_index_type_impl<_type> {                                    \
    constexpr static rocsparse_indextype value = _value;                       \
  }

MAP_ROCSPARSE_INDEX_TYPE(std::int32_t, rocsparse_indextype_i32);
MAP_ROCSPARSE_INDEX_TYPE(std::int64_t, rocsparse_indextype_i64);

#undef MAP_ROCSPARSE_INDEX_TYPE

} // namespace detail

/**
 * This is an alias for the `rocsparse_datatype` equivalent of `T`.
 *
 * @tparam T  a type
 *
 * @returns the actual `rocsparse_datatype`
 */
template <typename T>
constexpr rocsparse_datatype rocsparse_data_type() {
  return detail::rocsparse_data_type_impl<T>::value;
}

/**
 * This is an alias for the `rocsparse_indextype` equivalent of `T`.
 *
 * @tparam T  a type
 *
 * @returns the actual `rocsparse_indextype`
 */
template <typename T>
constexpr rocsparse_indextype rocsparse_index_type() {
  return detail::rocsparse_index_type_impl<T>::value;
}

} // namespace spblas
