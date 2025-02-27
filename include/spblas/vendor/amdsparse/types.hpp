#pragma once

#include <complex>
#include <cstdint>

#include <hipsparse/hipsparse.h>

namespace spblas {

using index_t = std::int64_t;
using offset_t = std::int64_t;

namespace detail {

/**
 * mapping the type to hipDataType_t
 */
template <typename T>
struct hip_data_type_impl {};

#define MAP_HIP_DATA_TYPE(_type, _value)                                      \
  template <>                                                                  \
  struct hip_data_type_impl<_type> {                                          \
    constexpr static hipDataType value = _value;                            \
  }

MAP_HIP_DATA_TYPE(float, HIP_R_32F);
MAP_HIP_DATA_TYPE(double, HIP_R_64F);
MAP_HIP_DATA_TYPE(std::complex<float>, HIP_C_32F);
MAP_HIP_DATA_TYPE(std::complex<double>, HIP_C_64F);
MAP_HIP_DATA_TYPE(std::int32_t, HIP_R_32I);
MAP_HIP_DATA_TYPE(std::int64_t, HIP_R_64I);

#undef MAP_HIP_DATA_TYPE

/**
 * mapping the type to hipsparseIndexType_t
 */
template <typename T>
struct hipsparse_index_type_impl {};

#define MAP_HIPSPARSE_INDEX_TYPE(_type, _value)                                 \
  template <>                                                                  \
  struct hipsparse_index_type_impl<_type> {                                     \
    constexpr static hipsparseIndexType_t value = _value;                       \
  }

MAP_HIPSPARSE_INDEX_TYPE(std::int32_t, HIPSPARSE_INDEX_32I);
MAP_HIPSPARSE_INDEX_TYPE(std::int64_t, HIPSPARSE_INDEX_64I);

#undef MAP_HIPSPARSE_INDEX_TYPE

} // namespace detail

/**
 * This is an alias for the `hipDataType` equivalent of `T`.
 *
 * @tparam T  a type
 *
 * @returns the actual `hipDataType`
 */
template <typename T>
constexpr hipDataType hip_data_type() {
  return detail::hip_data_type_impl<T>::value;
}

/**
 * This is an alias for the `hipIndexType_t` equivalent of `T`.
 *
 * @tparam T  a type
 *
 * @returns the actual `hipsparseIndexType_t`
 */
template <typename T>
constexpr hipsparseIndexType_t hipsparse_index_type() {
  return detail::hipsparse_index_type_impl<T>::value;
}

} // namespace spblas
