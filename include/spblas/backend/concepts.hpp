#pragma once

namespace spblas {

namespace __backend {

template <typename T>
concept row_iterable = requires(T& r) { rows(r); };

}

} // namespace spblas
