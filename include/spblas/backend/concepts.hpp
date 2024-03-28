#pragma once

#include <spblas/backend/cpos.hpp>
#include <spblas/detail/types.hpp>

namespace spblas {

namespace __backend {

template <typename T>
concept row_iterable = requires(T& t) { rows(t); };

template <typename T>
concept row_lookupable = requires(T& t) { lookup_row(t, tensor_index_t<T>{}); };

namespace {

template <typename T>
concept lookupable_matrix =
    requires(T& t, tensor_index_t<T> i, tensor_index_t<T> j) {
      { lookup(t, i, j) };
    };

template <typename T>
concept lookupable_vector = requires(T& t, tensor_index_t<T> i) {
  { lookup(t, i) };
};

} // namespace

template <typename T>
concept lookupable = lookupable_matrix<T> || lookupable_vector<T>;

} // namespace __backend

} // namespace spblas
