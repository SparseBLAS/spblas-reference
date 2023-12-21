#pragma once

#include <spblas/backend/concepts.hpp>
#include <spblas/backend/cpos.hpp>
#include <spblas/views/views.hpp>

namespace spblas {

namespace __backend {

template <matrix M, typename F>
  requires(__backend::row_iterable<M>)
void for_each(M&& m, F&& f) {
  for (auto&& [i, row] : __backend::rows(m)) {
    for (auto&& [j, v] : row) {
      f(std::make_tuple(std::tuple{i, j}, std::reference_wrapper(v)));
    }
  }
}

template <vector V, typename F>
  requires(__backend::lookupable<V> && __ranges::random_access_range<V>)
void for_each(V&& v, F&& f) {
  using index_type = __ranges::range_size_t<V>;
  for (index_type i = 0; i < __backend::shape(v); i++) {
    auto&& value = __backend::lookup(v, i);
    f(std::make_tuple(i, std::reference_wrapper(value)));
  }
}

} // namespace __backend

} // namespace spblas
