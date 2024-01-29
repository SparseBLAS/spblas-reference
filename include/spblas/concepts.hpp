#pragma once

#include <concepts>
#include <spblas/detail/ranges.hpp>
#include <spblas/views/inspectors.hpp>
#include <spblas/views/view_base.hpp>

namespace spblas {

template <typename M>
concept matrix = __detail::is_csr_view_v<M> ||
                 __detail::is_matrix_instantiation_of_mdspan_v<M>;

template <typename V>
concept vector = __ranges::random_access_range<V> && !matrix<V>;

template <typename T>
concept tensor = matrix<T> || vector<T>;

template <typename T>
concept view = tensor<T> && (std::derived_from<T, view_base> ||
                             __detail::is_matrix_instantiation_of_mdspan_v<T> ||
                             __detail::view<T>);

} // namespace spblas
