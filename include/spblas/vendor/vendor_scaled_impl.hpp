#pragma once

#include <spblas/concepts.hpp>

namespace spblas {

// XXX this only works in the absence of a reference version (ref only has vector `scaled` currently)
template <typename Scalar, matrix M>
auto scaled(Scalar alpha, M&& m) {
  return vendor_csr_view(alpha, m);
}

} // namespace spblas
