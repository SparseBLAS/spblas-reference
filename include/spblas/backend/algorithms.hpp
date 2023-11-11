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

} // namespace __backend

} // namespace spblas
