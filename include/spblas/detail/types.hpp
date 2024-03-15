#pragma once

#include <cstddef>
#include <type_traits>

#ifdef ENABLE_ONEMKL
#include <spblas/vendor/mkl/types.hpp>
#endif

namespace spblas {

#ifndef ENABLE_ONEMKL
using index_t = std::size_t;
#endif

template <typename T>
struct tensor_traits {
  using scalar_type = typename std::remove_cvref_t<T>::scalar_type;
  using scalar_reference = typename std::remove_cvref_t<T>::scalar_reference;
  using index_type = typename std::remove_cvref_t<T>::index_type;
  using offset_type = typename std::remove_cvref_t<T>::offset_type;
};

template <typename T>
using tensor_scalar_t = typename tensor_traits<T>::scalar_type;

template <typename T>
using tensor_scalar_reference_t = typename tensor_traits<T>::scalar_reference;

template <typename T>
using tensor_index_t = typename tensor_traits<T>::index_type;

template <typename T>
using tensor_offset_t = typename tensor_traits<T>::offset_type;

} // namespace spblas
