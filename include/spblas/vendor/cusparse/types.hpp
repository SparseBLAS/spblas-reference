#pragma once

#include <complex>
#include <cstdint>
#include <type_traits>

#include <cuda.h>
#include <cusparse.h>

namespace spblas {

using index_t = std::int32_t;
using offset_t = index_t;

namespace detail {

template <typename T>
constexpr static bool is_valid_cusparse_scalar_type_v =
    std::is_floating_point_v<T> || std::is_same_v<T, std::int8_t> ||
    std::is_same_v<T, std::int32_t>;

template <typename T>
constexpr static bool is_valid_cusparse_index_type_v =
    std::is_same_v<T, std::int32_t> || std::is_same_v<T, std::int64_t>;

template <typename T>
struct cuda_data_type;

template <>
struct cuda_data_type<float> {
  constexpr static cudaDataType_t value = CUDA_R_32F;
};

template <>
struct cuda_data_type<double> {
  constexpr static cudaDataType_t value = CUDA_R_64F;
};

template <>
struct cuda_data_type<std::complex<float>> {
  constexpr static cudaDataType_t value = CUDA_C_32F;
};

template <>
struct cuda_data_type<std::complex<double>> {
  constexpr static cudaDataType_t value = CUDA_C_64F;
};

template <>
struct cuda_data_type<std::int8_t> {
  constexpr static cudaDataType_t value = CUDA_R_8I;
};

template <>
struct cuda_data_type<std::int32_t> {
  constexpr static cudaDataType_t value = CUDA_R_32I;
};

template <typename T>
constexpr static cudaDataType_t cuda_data_type_v = cuda_data_type<T>::value;

template <typename T>
struct cuda_index_type;

template <>
struct cuda_index_type<std::int32_t> {
  constexpr static cusparseIndexType_t value = CUSPARSE_INDEX_32I;
};

template <>
struct cuda_index_type<std::int64_t> {
  constexpr static cusparseIndexType_t value = CUSPARSE_INDEX_64I;
};

template <typename T>
constexpr static cusparseIndexType_t cusparse_index_type_v =
    cuda_index_type<T>::value;

} // namespace detail

} // namespace spblas
