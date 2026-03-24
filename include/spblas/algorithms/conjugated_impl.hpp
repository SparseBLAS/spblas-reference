#pragma once

#include <complex>
#include <utility>

#include <spblas/concepts.hpp>
#include <spblas/detail/type_traits.hpp>
#include <spblas/views/conjugated_view.hpp>

namespace spblas {

template <vector V>
auto conjugated(V&& v) {
  if constexpr (__detail::is_std_complex_v<tensor_scalar_t<V>>) {
    return conjugated_view(std::forward<V>(v));
  } else {
    return std::forward<V>(v);
  }
}

template <matrix M>
auto conjugated(M&& m) {
  if constexpr (__detail::is_std_complex_v<tensor_scalar_t<M>>) {
    return conjugated_view(std::forward<M>(m));
  } else {
    return std::forward<M>(m);
  }
}

} // namespace spblas
