#pragma once

#include <spblas/views/views.hpp>

namespace spblas {

template <typename M>
concept matrix = __detail::is_csr_view_v<M> ||
                 __detail::is_matrix_instantiation_of_mdspan_v<M>;

template <typename V>
concept vector = __ranges::random_access_range<V> && !matrix<V>;

} // namespace spblas
