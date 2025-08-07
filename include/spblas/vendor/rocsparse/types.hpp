#pragma once

#include <complex>
#include <cstdint>
#include <type_traits>

#include <rocsparse/rocsparse.h>

namespace spblas {

using index_t = std::int32_t;
using offset_t = index_t;

namespace detail {

template <typename T>
constexpr static bool is_valid_rocsparse_scalar_type_v =
    std::is_same_v<T, std::int32_t> || std::is_same_v<T, std::uint32_t> ||
    std::is_floating_point_v<T>;

template <typename T>
constexpr static bool is_valid_rocsparse_index_type_v =
    std::is_same_v<T, std::uint16_t> || std::is_same_v<T, std::int32_t> ||
    std::is_same_v<T, std::int64_t>;

template <typename T>
struct rocsparse_data_type;

template <>
struct rocsparse_data_type<std::int32_t> {
  constexpr static rocsparse_datatype value = rocsparse_datatype_i32_r;
};

template <>
struct rocsparse_data_type<std::uint32_t> {
  constexpr static rocsparse_datatype value = rocsparse_datatype_u32_r;
};

template <>
struct rocsparse_data_type<float> {
  constexpr static rocsparse_datatype value = rocsparse_datatype_f32_r;
};

template <>
struct rocsparse_data_type<double> {
  constexpr static rocsparse_datatype value = rocsparse_datatype_f64_r;
};

template <>
struct rocsparse_data_type<std::complex<float>> {
  constexpr static rocsparse_datatype value = rocsparse_datatype_f32_c;
};

template <>
struct rocsparse_data_type<std::complex<double>> {
  constexpr static rocsparse_datatype value = rocsparse_datatype_f64_c;
};

template <typename T>
constexpr static rocsparse_datatype rocsparse_data_type_v =
    rocsparse_data_type<T>::value;

template <typename T>
struct rocsparse_index_type;

template <>
struct rocsparse_index_type<std::uint16_t> {
  constexpr static rocsparse_indextype value = rocsparse_indextype_u16;
};

template <>
struct rocsparse_index_type<std::int32_t> {
  constexpr static rocsparse_indextype value = rocsparse_indextype_i32;
};

template <>
struct rocsparse_index_type<std::int64_t> {
  constexpr static rocsparse_indextype value = rocsparse_indextype_i64;
};

template <typename T>
constexpr static rocsparse_indextype rocsparse_index_type_v =
    rocsparse_index_type<T>::value;

} // namespace detail

} // namespace spblas
