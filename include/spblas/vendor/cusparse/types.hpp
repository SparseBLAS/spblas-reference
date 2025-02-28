#pragma once

#include <complex>
#include <cstdint>

#include <cuda.h>
#include <cusparse.h>

namespace spblas {

using index_t = std::int32_t;
using offset_t = index_t;

namespace detail {

/**
 * mapping the type to cudaDataType_t
 */
template <typename T>
struct cuda_datatype_traits {};

#define MAP_CUDA_DATATYPE(_type, _value)                                       \
  template <>                                                                  \
  struct cuda_datatype_traits<_type> {                                         \
    constexpr static cudaDataType_t value = _value;                            \
  }

MAP_CUDA_DATATYPE(float, CUDA_R_32F);
MAP_CUDA_DATATYPE(double, CUDA_R_64F);
MAP_CUDA_DATATYPE(std::complex<float>, CUDA_C_32F);
MAP_CUDA_DATATYPE(std::complex<double>, CUDA_C_64F);

#undef MAP_CUDA_DATATYPE

/**
 * mapping the type to cusparseIndexType_t
 */
template <typename T>
struct cusparse_indextype_traits {};

#define MAP_CUSPARSE_INDEXTYPE(_type, _value)                                  \
  template <>                                                                  \
  struct cusparse_indextype_traits<_type> {                                    \
    constexpr static cusparseIndexType_t value = _value;                       \
  }

MAP_CUSPARSE_INDEXTYPE(std::int32_t, CUSPARSE_INDEX_32I);
MAP_CUSPARSE_INDEXTYPE(std::int64_t, CUSPARSE_INDEX_64I);

#undef MAP_CUSPARSE_INDEXTYPE

} // namespace detail

/**
 * This is an alias for the `cudaDataType_t` equivalent of `T`.
 *
 * @tparam T  a type
 *
 * @returns the actual `cudaDataType_t`
 */
template <typename T>
constexpr cudaDataType_t to_cuda_datatype() {
  return detail::cuda_datatype_traits<T>::value;
}

/**
 * This is an alias for the `cudaIndexType_t` equivalent of `T`.
 *
 * @tparam T  a type
 *
 * @returns the actual `cusparseIndexType_t`
 */
template <typename T>
constexpr cusparseIndexType_t to_cusparse_indextype() {
  return detail::cusparse_indextype_traits<T>::value;
}

} // namespace spblas
