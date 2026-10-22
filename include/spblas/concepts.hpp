#pragma once

#include <concepts>
#include <spblas/detail/concepts.hpp>
#include <spblas/detail/ranges.hpp>
#include <spblas/views/inspectors.hpp>
#include <spblas/views/view_base.hpp>

namespace spblas {

/*
  The following types fulfill the matrix concept:
  - Instantiations of csr_view<...>
  - Instantiations of csc_view<...>
  - Instantiations of mdspan<...> with rank 2
  - Instantiations of scaled_view<T> where M is a matrix
  - Instantiations of conjugated_view<T> where M is a matrix
*/

template <typename M>
concept matrix = __detail::is_csr_view_v<M> || __detail::is_csc_view_v<M> ||
                 __detail::is_matrix_mdspan_v<M> || __detail::matrix<M>;

/*
  The following types fulfill the vector concept:
  - Random access range (e.g. std::vector<...>)
*/

template <typename V>
concept vector = __ranges::random_access_range<V> && !matrix<V>;

template <typename T>
concept tensor = matrix<T> || vector<T>;

namespace __detail {

// Lives in __detail because `spblas::view` is the namespace of view
// functions (`spblas::view::transpose`, `spblas::view::triangle`, ...).
template <typename T>
concept view =
    tensor<T> && (std::derived_from<std::remove_cvref_t<T>, view_base> ||
                  is_matrix_mdspan_v<T> || __ranges::view<T>);

} // namespace __detail

} // namespace spblas
