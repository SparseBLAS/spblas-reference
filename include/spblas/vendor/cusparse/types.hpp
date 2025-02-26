#pragma once

#include <complex>
#include <cstdint>

#include <cuda.h>
#include <cusparse.h>

namespace spblas {

using index_t = std::int64_t;
using offset_t = std::int64_t;

namespace detail {

/**
 * mapping the type to cudaDataType_t
 */
template <typename T>
struct cuda_data_type_impl {};

#define MAP_CUDA_DATA_TYPE(_type, _value)                                      \
  template <>                                                                  \
  struct cuda_data_type_impl<_type> {                                          \
    constexpr static cudaDataType_t value = _value;                            \
  }

MAP_CUDA_DATA_TYPE(float, CUDA_R_32F);
MAP_CUDA_DATA_TYPE(double, CUDA_R_64F);
MAP_CUDA_DATA_TYPE(std::complex<float>, CUDA_C_32F);
MAP_CUDA_DATA_TYPE(std::complex<double>, CUDA_C_64F);
MAP_CUDA_DATA_TYPE(std::int32_t, CUDA_R_32I);
MAP_CUDA_DATA_TYPE(std::int64_t, CUDA_R_64I);

#undef MAP_CUDA_DATA_TYPE

/**
 * mapping the type to cusparseIndexType_t
 */
template <typename T>
struct cusparse_index_type_impl {};

#define MAP_CUSPARSE_INDEX_TYPE(_type, _value)                                 \
  template <>                                                                  \
  struct cusparse_index_type_impl<_type> {                                     \
    constexpr static cusparseIndexType_t value = _value;                       \
  }

MAP_CUSPARSE_INDEX_TYPE(std::int32_t, CUSPARSE_INDEX_32I);
MAP_CUSPARSE_INDEX_TYPE(std::int64_t, CUSPARSE_INDEX_64I);

#undef MAP_CUSPARSE_INDEX_TYPE

} // namespace detail

/**
 * This is an alias for the `cudaDataType_t` equivalent of `T`.
 *
 * @tparam T  a type
 *
 * @returns the actual `cudaDataType_t`
 */
template <typename T>
constexpr cudaDataType_t cuda_data_type() {
  return detail::cuda_data_type_impl<T>::value;
}

/**
 * This is an alias for the `cudaIndexType_t` equivalent of `T`.
 *
 * @tparam T  a type
 *
 * @returns the actual `cusparseIndexType_t`
 */
template <typename T>
constexpr cusparseIndexType_t cusparse_index_type() {
  return detail::cusparse_index_type_impl<T>::value;
}

} // namespace spblas
