#pragma once

#include <spblas/views/views.hpp>

namespace spblas {

template <typename M>
concept matrix = __detail::is_csr_view_v<M>;

template <typename V>
concept vector = __ranges::contiguous_range<V>;

} // namespace spblas
