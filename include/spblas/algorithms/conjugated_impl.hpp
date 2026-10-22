#pragma once

#include <complex>
#include <utility>

#include <spblas/concepts.hpp>
#include <spblas/detail/type_traits.hpp>
#include <spblas/views/conjugated_view.hpp>
#include <spblas/views/inspectors.hpp>

namespace spblas {

// conj(conj(x)) is x. (Spelled out: conjugated_view(v) of a conjugated_view
// would deduce a copy of it, through the copy deduction candidate.)
template <vector V>
auto conjugated(V&& v) {
  if constexpr (__detail::is_conjugated_view_v<V>) {
    return v.base();
  } else if constexpr (__detail::is_std_complex_v<tensor_scalar_t<V>>) {
    return conjugated_view(std::forward<V>(v));
  } else {
    return std::forward<V>(v);
  }
}

template <matrix M>
auto conjugated(M&& m) {
  if constexpr (__detail::is_conjugated_view_v<M>) {
    return m.base();
  } else if constexpr (__detail::is_std_complex_v<tensor_scalar_t<M>>) {
    return conjugated_view(std::forward<M>(m));
  } else {
    return std::forward<M>(m);
  }
}

} // namespace spblas
