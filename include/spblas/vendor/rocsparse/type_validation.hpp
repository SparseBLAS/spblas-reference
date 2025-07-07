#pragma once

#include <spblas/detail/types.hpp>
#include <spblas/vendor/rocsparse/types.hpp>

namespace spblas {
namespace detail {

template <typename T>
static constexpr bool has_valid_rocsparse_matrix_types_v =
    is_valid_rocsparse_scalar_type_v<tensor_scalar_t<T>> &&
    is_valid_rocsparse_index_type_v<tensor_index_t<T>> &&
    is_valid_rocsparse_index_type_v<tensor_offset_t<T>>;

template <typename T>
static constexpr bool has_valid_rocsparse_vector_types_v =
    is_valid_rocsparse_scalar_type_v<tensor_scalar_t<T>>;

} // namespace detail
} // namespace spblas
