#pragma once

#include <spblas/backend/cpos.hpp>
#include <spblas/concepts.hpp>
#include <spblas/detail/types.hpp>

namespace spblas {

namespace __backend {

template <typename T>
concept row_iterable = requires(T& r) { rows(r); };

namespace {

template <typename T>
concept lookupable_matrix =
    spblas::matrix<T> &&
    requires(T& t, tensor_index_t<T> i, tensor_index_t<T> j) {
      { lookup(t, i, j) };
    };

template <typename T>
concept lookupable_vector =
    spblas::vector<T> && requires(T& t, tensor_index_t<T> i) {
      { lookup(t, i) };
    };

} // namespace

template <typename T>
concept lookupable = lookupable_matrix<T> || lookupable_vector<T>;

} // namespace __backend

} // namespace spblas
