#pragma once

#include <spblas/backend/backend.hpp>
#include <spblas/concepts.hpp>
#include <spblas/detail/log.hpp>

#include <spblas/algorithms/transposed.hpp>
#include <spblas/backend/csr_builder.hpp>
#include <spblas/backend/spa_accumulator.hpp>
#include <spblas/detail/operation_info_t.hpp>

#include <algorithm>

namespace spblas {

// C = AB
// SpMV
template <matrix A, vector B, vector C>
  requires(__backend::lookupable<B> && __backend::lookupable<C>)
void multiply(A&& a, B&& b, C&& c) {
  log_trace("");
  if (__backend::shape(a)[0] != __backend::shape(c) ||
      __backend::shape(a)[1] != __backend::shape(b)) {
    throw std::invalid_argument(
        "multiply: matrix and vector dimensions are incompatible.");
  }

  __backend::for_each(c, [](auto&& e) {
    auto&& [_, v] = e;
    v = 0;
  });

  __backend::for_each(a, [&](auto&& e) {
    auto&& [idx, a_v] = e;
    auto&& [i, k] = idx;
    __backend::lookup(c, i) += a_v * __backend::lookup(b, k);
  });
}

// C = AB
// SpMM
template <matrix A, matrix B, matrix C>
  requires(__backend::lookupable<B> && __backend::lookupable<C>)
void multiply(A&& a, B&& b, C&& c) {
  log_trace("");
  if (__backend::shape(a)[0] != __backend::shape(c)[0] ||
      __backend::shape(b)[1] != __backend::shape(c)[1] ||
      __backend::shape(a)[1] != __backend::shape(b)[0]) {
    throw std::invalid_argument(
        "multiply: matrix dimensions are incompatible.");
  }

  __backend::for_each(c, [](auto&& e) {
    auto&& [_, v] = e;
    v = 0;
  });

  __backend::for_each(a, [&](auto&& e) {
    auto&& [idx, a_v] = e;
    auto&& [i, k] = idx;
    for (std::size_t j = 0; j < __backend::shape(b)[1]; j++) {
      __backend::lookup(c, i, j) += a_v * __backend::lookup(b, k, j);
    }
  });
}

// C = AB
// SpGEMM (Gustavson's Algorithm)
template <matrix A, matrix B, matrix C>
  requires(__backend::row_iterable<A> && __backend::row_iterable<B> &&
           __detail::is_csr_view_v<C>)
void multiply(A&& a, B&& b, C&& c) {
  log_trace("");
  if (__backend::shape(a)[0] != __backend::shape(c)[0] ||
      __backend::shape(b)[1] != __backend::shape(c)[1] ||
      __backend::shape(a)[1] != __backend::shape(b)[0]) {
    throw std::invalid_argument(
        "multiply: matrix dimensions are incompatible.");
  }

  using T = tensor_scalar_t<C>;
  using I = tensor_index_t<C>;

  __backend::spa_accumulator<T, I> c_row(__backend::shape(c)[1]);
  __backend::csr_builder c_builder(c);

  for (auto&& [i, a_row] : __backend::rows(a)) {
    c_row.clear();
    for (auto&& [k, a_v] : a_row) {
      for (auto&& [j, b_v] : __backend::lookup_row(b, k)) {
        c_row[j] += a_v * b_v;
      }
    }
    c_row.sort();

    try {
      c_builder.insert_row(i, c_row.get());
    } catch (...) {
      throw std::runtime_error("multiply: SpGEMM ran out of memory.");
    }
  }
  c.update(c.values(), c.rowptr(), c.colind(), c.shape(),
           c.rowptr()[c.shape()[0]]);
}

template <matrix A, matrix B, matrix C>
  requires(__backend::column_iterable<A> && __backend::column_iterable<B> &&
           __detail::is_csc_view_v<C>)
void multiply(A&& a, B&& b, C&& c) {
  log_trace("");
  if (__backend::shape(a)[0] != __backend::shape(c)[0] ||
      __backend::shape(b)[1] != __backend::shape(c)[1] ||
      __backend::shape(a)[1] != __backend::shape(b)[0]) {
    throw std::invalid_argument(
        "multiply: matrix dimensions are incompatible.");
  }
  multiply(transposed(b), transposed(a), transposed(c));
}

template <matrix A, matrix B, matrix C>
operation_info_t multiply_inspect(A&& a, B&& b, C&& c) {
  return operation_info_t{};
}

template <matrix A, matrix B, matrix C>
void multiply_inspect(operation_info_t& info, A&& a, B&& b, C&& c){};

// C = AB
// SpGEMM (Gustavson's Algorithm)
template <matrix A, matrix B, matrix C>
  requires(__backend::row_iterable<A> && __backend::row_iterable<B> &&
           __detail::is_csr_view_v<C>)
operation_info_t multiply_compute(A&& a, B&& b, C&& c) {
  log_trace("");
  if (__backend::shape(a)[0] != __backend::shape(c)[0] ||
      __backend::shape(b)[1] != __backend::shape(c)[1] ||
      __backend::shape(a)[1] != __backend::shape(b)[0]) {
    throw std::invalid_argument(
        "multiply: matrix dimensions are incompatible.");
  }

  using T = tensor_scalar_t<C>;
  using I = tensor_index_t<C>;
  using O = tensor_offset_t<C>;

  O nnz = 0;
  __backend::spa_set<I> c_row(__backend::shape(c)[1]);

  for (auto&& [i, a_row] : __backend::rows(a)) {
    c_row.clear();

    for (auto&& [k, a_v] : a_row) {
      for (auto&& [j, b_v] : __backend::lookup_row(b, k)) {
        c_row.insert(j);
      }
    }

    nnz += c_row.size();
  }

  return operation_info_t{__backend::shape(c), nnz};
}

// C = AB
// SpGEMM (Gustavson's Algorithm, transposed)
template <matrix A, matrix B, matrix C>
  requires(__backend::column_iterable<A> && __backend::column_iterable<B> &&
           __detail::is_csc_view_v<C>)
operation_info_t multiply_compute(A&& a, B&& b, C&& c) {
  log_trace("");
  if (__backend::shape(a)[0] != __backend::shape(c)[0] ||
      __backend::shape(b)[1] != __backend::shape(c)[1] ||
      __backend::shape(a)[1] != __backend::shape(b)[0]) {
    throw std::invalid_argument(
        "multiply: matrix dimensions are incompatible.");
  }

  auto info = multiply_compute(transposed(b), transposed(a), transposed(c));
  info.update_impl_({info.result_shape()[1], info.result_shape()[0]},
                    info.result_nnz());
  return info;
}

template <matrix A, matrix B, matrix C>
  requires(__backend::row_iterable<A> && __backend::row_iterable<B> &&
           __detail::is_csr_view_v<C>)
void multiply_compute(operation_info_t& info, A&& a, B&& b, C&& c) {
  auto new_info = multiply_compute(std::forward<A>(a), std::forward<B>(b),
                                   std::forward<C>(c));
  info.update_impl_(new_info.result_shape(), new_info.result_nnz());
}

template <typename T, typename A, typename B>
std::optional<T> sparse_dot_product(A&& a, B&& b) {
  auto sort_by_index = [](auto&& a, auto&& b) {
    auto&& [a_i, a_v] = a;
    auto&& [b_i, b_v] = b;
    return a_i < b_i;
  };
  std::sort(a.begin(), a.end(), sort_by_index);
  std::sort(b.begin(), b.end(), sort_by_index);

  auto a_iter = a.begin();
  auto b_iter = b.begin();

  T sum = 0;
  bool implicit_zero = true;
  for (; a_iter != a.end() && b_iter != b.end();) {
    auto&& [a_i, a_v] = *a_iter;
    auto&& [b_i, b_v] = *b_iter;

    if (a_i == b_i) {
      sum += a_v * b_v;
      implicit_zero = false;
    } else if (a_i < b_i) {
      ++a_iter;
    } else {
      ++b_iter;
    }
  }

  if (implicit_zero) {
    return {};
  } else {
    return sum;
  }
}

// C = AB
// SpGEMM (Inner Product)
template <matrix A, matrix B, matrix C>
  requires(__backend::row_iterable<A> && __backend::column_iterable<B> &&
           __detail::is_csr_view_v<C>)
void multiply(A&& a, B&& b, C&& c) {
  log_trace("");
  if (__backend::shape(a)[0] != __backend::shape(c)[0] ||
      __backend::shape(b)[1] != __backend::shape(c)[1] ||
      __backend::shape(a)[1] != __backend::shape(b)[0]) {
    throw std::invalid_argument(
        "multiply: matrix dimensions are incompatible.");
  }

  using T = tensor_scalar_t<C>;
  using I = tensor_index_t<C>;

  __backend::spa_accumulator<T, I> c_row(__backend::shape(c)[1]);
  __backend::csr_builder c_builder(c);

  for (auto&& [i, a_row] : __backend::rows(a)) {
    c_row.clear();

    if (!__ranges::empty(a_row)) {
      for (auto&& [j, b_column] : __backend::columns(b)) {
        if (!__ranges::empty(b_column)) {
          auto v = sparse_dot_product<T>(a_row, b_column);

          if (v.has_value()) {
            c_row[j] += v.value();
          }
        }
      }
      c_row.sort();

      try {
        c_builder.insert_row(i, c_row.get());
      } catch (...) {
        throw std::runtime_error("multiply: SpGEMM ran out of memory.");
      }
    }
  }
  c.update(c.values(), c.rowptr(), c.colind(), c.shape(),
           c.rowptr()[c.shape()[0]]);
}

// C = AB
// SpGEMM (Inner Product)
template <matrix A, matrix B, matrix C>
  requires(__backend::row_iterable<A> && __backend::column_iterable<B> &&
           __detail::is_csr_view_v<C>)
operation_info_t multiply_compute(A&& a, B&& b, C&& c) {
  log_trace("");
  if (__backend::shape(a)[0] != __backend::shape(c)[0] ||
      __backend::shape(b)[1] != __backend::shape(c)[1] ||
      __backend::shape(a)[1] != __backend::shape(b)[0]) {
    throw std::invalid_argument(
        "multiply: matrix dimensions are incompatible.");
  }

  using T = tensor_scalar_t<C>;
  using I = tensor_index_t<C>;
  using O = tensor_offset_t<C>;

  O nnz = 0;

  for (auto&& [i, a_row] : __backend::rows(a)) {
    if (!__ranges::empty(a_row)) {
      for (auto&& [j, b_column] : __backend::columns(b)) {
        if (!__ranges::empty(b_column)) {
          auto v = sparse_dot_product<T>(a_row, b_column);

          if (v.has_value()) {
            nnz++;
          }
        }
      }
    }
  }

  return operation_info_t{__backend::shape(c), nnz};
}

template <matrix A, matrix B, matrix C>
  requires(__backend::column_iterable<A> && __backend::column_iterable<B> &&
           __detail::is_csc_view_v<C>)
void multiply_compute(operation_info_t& info, A&& a, B&& b, C&& c) {
  auto new_info = multiply_compute(std::forward<A>(a), std::forward<B>(b),
                                   std::forward<C>(c));
  info.update_impl_(new_info.result_shape(), new_info.result_nnz());
}

// C = AB
template <matrix A, matrix B, matrix C>
void multiply_fill(operation_info_t info, A&& a, B&& b, C&& c) {
  log_trace("");
  multiply(a, b, c);
}

} // namespace spblas
