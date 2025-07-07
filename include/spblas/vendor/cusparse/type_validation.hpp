#pragma once

#include <spblas/detail/types.hpp>
#include <spblas/vendor/cusparse/types.hpp>

namespace spblas {
namespace detail {

template <typename T>
static constexpr bool has_valid_cusparse_matrix_types_v =
    is_valid_cusparse_scalar_type_v<tensor_scalar_t<T>> &&
    is_valid_cusparse_index_type_v<tensor_index_t<T>> &&
    is_valid_cusparse_index_type_v<tensor_offset_t<T>>;

template <typename T>
static constexpr bool has_valid_cusparse_vector_types_v =
    is_valid_cusparse_scalar_type_v<tensor_scalar_t<T>>;

} // namespace detail
} // namespace spblas
