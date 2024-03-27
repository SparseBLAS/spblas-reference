#pragma once

#include <spblas/backend/cpos.hpp>

#include <any>
#include <concepts>
#include <iterator>

#include <spblas/detail/tuple_concept.hpp>

namespace spblas {

namespace __detail {

template <typename M>
concept matrix = requires(M& m) {
  { __backend::size(m) } -> std::weakly_incrementable;
  { __backend::shape(m) } -> tuple_like<std::size_t, std::size_t>;
};

}

} // namespace spblas
