#pragma once

#include <spblas/backend/backend.hpp>
#include <spblas/concepts.hpp>
#include <spblas/detail/log.hpp>

#include <spblas/algorithms/transposed.hpp>
#include <spblas/backend/csr_builder.hpp>
#include <spblas/backend/spa_accumulator.hpp>
#include <spblas/detail/api_concepts.hpp>
#include <spblas/detail/operation_info_t.hpp>
#include <spblas/detail/operation_state.hpp>
#include <spblas/views/inspectors.hpp>

#include "detail/spgemm/spgemm.hpp"

#include <algorithm>

namespace spblas {

// SpMV inspect
template <matrix A, vector B, vector C>
operation_info_t multiply_inspect(A&& a, B&& b, C&& c) {
  log_trace("");
  return operation_info_t{};
}

// SpMV inspect
template <matrix A, vector B, vector C>
void multiply_inspect(operation_info_t& info, A&& a, B&& b, C&& c) {
  log_trace("");
}

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
// SpMV with info input
template <matrix A, vector B, vector C>
  requires(__backend::lookupable<B> && __backend::lookupable<C>)
void multiply(operation_info_t& info, A&& a, B&& b, C&& c) {
  log_trace("");
  multiply(std::forward<A>(a), std::forward<B>(b), std::forward<C>(c));
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

  // initializes c to zero so we can use += everywhere
  __backend::for_each(c, [](auto&& e) {
    auto&& [_, v] = e;
    v = 0;
  });

  // traverses elements of a and performs appropriate
  // multiplication with B rows
  __backend::for_each(a, [&](auto&& e) {
    auto&& [idx, a_v] = e;
    auto&& [i, k] = idx;
    for (std::size_t j = 0; j < __backend::shape(b)[1]; j++) {
      __backend::lookup(c, i, j) += a_v * __backend::lookup(b, k, j);
    }
  });
}

// C = AB
// SpMM with info
template <matrix A, matrix B, matrix C>
  requires(__backend::lookupable<B> && __backend::lookupable<C>)
void multiply(operation_info_t& info, A&& a, B&& b, C&& c) {
  log_trace("");
  multiply(std::forward<A>(a), std::forward<B>(b), std::forward<C>(c));
}

// C = AB
// SpMM or SpGEMM multiply_inspect variants end up here
template <matrix A, matrix B, matrix C>
operation_info_t multiply_inspect(A&& a, B&& b, C&& c) {
  log_trace("");
  return operation_info_t{};
}

// C = AB
// SpMM or SpGEMM multiply_inspect variants end up here
template <matrix A, matrix B, matrix C>
void multiply_inspect(operation_info_t& info, A&& a, B&& b, C&& c) {
  log_trace("");
};

// C = AB
// SpGEMM compute stage with CSR output
template <matrix A, matrix B, matrix C>
  requires(__backend::row_iterable<A> && __backend::row_iterable<B> &&
           __detail::is_csr_view_v<C>)
void multiply_compute(operation_info_t& info, A&& a, B&& b, C&& c) {
  log_trace("");
  auto new_info = multiply_compute(std::forward<A>(a), std::forward<B>(b),
                                   std::forward<C>(c));
  info.update_impl_(new_info.result_shape(), new_info.result_nnz());
}

// C = AB
// SpGEMM compute stage with CSC output
template <matrix A, matrix B, matrix C>
  requires(__backend::column_iterable<A> && __backend::column_iterable<B> &&
           __detail::is_csc_view_v<C>)
void multiply_compute(operation_info_t& info, A&& a, B&& b, C&& c) {
  auto new_info = multiply_compute(std::forward<A>(a), std::forward<B>(b),
                                   std::forward<C>(c));
  info.update_impl_(new_info.result_shape(), new_info.result_nnz());
}

// C = AB
// SpGEMM fill stage with CSR or CSC output
template <matrix A, matrix B, matrix C>
void multiply_fill(operation_info_t info, A&& a, B&& b, C&& c) {
  log_trace("");
  multiply(a, b, c);
}

//
// operation_state API (SparseBLAS paper). Built on the kernels above.
//
//   multiply_inspect([policy,] state, alpha, A, x, [beta, z,] y)
//   multiply([policy,] state, alpha, A, x, [beta, z,] y)
//     SpMV / SpMM: y = alpha * A * x [+ beta * z]. z may alias y; if
//     beta == 0, z is not read.
//
//   multiply_inspect([policy,] state, alpha, A, B, C)
//   multiply_compute([policy,] state, alpha, A, B, C)
//   multiply_fill([policy,] state, alpha, A, B, C)
//     SpGEMM: C = alpha * A * B. multiply_compute sets
//     state.get_result_nnz(); the user then attaches arrays of that size to
//     C, and multiply_fill fills them.
//

namespace __detail {

// y = beta * z, or y = 0 if beta == 0 (z is not read then).
template <typename Beta, tensor Z, tensor Y>
void assign_scaled_(Beta beta, Z&& z, Y&& y) {
  using V = tensor_scalar_t<Y>;
  const V beta_v = static_cast<V>(beta);
  const bool zero = beta_v == V(0);

  if constexpr (vector<Y>) {
    for (tensor_index_t<Y> i = 0; i < __backend::shape(y); i++) {
      __backend::lookup(y, i) = zero ? V(0) : beta_v * __backend::lookup(z, i);
    }
  } else {
    for (tensor_index_t<Y> i = 0; i < __backend::shape(y)[0]; i++) {
      for (tensor_index_t<Y> j = 0; j < __backend::shape(y)[1]; j++) {
        __backend::lookup(y, i, j) =
            zero ? V(0) : beta_v * __backend::lookup(z, i, j);
      }
    }
  }
}

} // namespace __detail

// SpMV: y = alpha * A * x + beta * z
template <__detail::scalar Alpha, matrix A, vector X, __detail::scalar Beta,
          vector Z, vector Y>
  requires(__backend::lookupable<X> && __backend::lookupable<Z> &&
           __backend::lookupable<Y>)
void multiply(operation_state& state, Alpha alpha, A&& a, X&& x, Beta beta,
              Z&& z, Y&& y) {
  log_trace("");
  if (__backend::shape(a)[0] != __backend::shape(y) ||
      __backend::shape(a)[1] != __backend::shape(x) ||
      __backend::shape(z) != __backend::shape(y)) {
    throw std::invalid_argument(
        "multiply: matrix and vector dimensions are incompatible.");
  }

  using V = tensor_scalar_t<Y>;
  const V alpha_v = static_cast<V>(alpha);

  __detail::assign_scaled_(beta, z, y);

  __backend::for_each(a, [&](auto&& e) {
    auto&& [idx, a_v] = e;
    auto&& [i, k] = idx;
    __backend::lookup(y, i) += alpha_v * (a_v * __backend::lookup(x, k));
  });
}

// SpMM: Y = alpha * A * X + beta * Z
template <__detail::scalar Alpha, matrix A, matrix X, __detail::scalar Beta,
          matrix Z, matrix Y>
  requires(__backend::lookupable<X> && __backend::lookupable<Z> &&
           __backend::lookupable<Y>)
void multiply(operation_state& state, Alpha alpha, A&& a, X&& x, Beta beta,
              Z&& z, Y&& y) {
  log_trace("");
  if (__backend::shape(a)[0] != __backend::shape(y)[0] ||
      __backend::shape(x)[1] != __backend::shape(y)[1] ||
      __backend::shape(a)[1] != __backend::shape(x)[0] ||
      __backend::shape(z)[0] != __backend::shape(y)[0] ||
      __backend::shape(z)[1] != __backend::shape(y)[1]) {
    throw std::invalid_argument(
        "multiply: matrix dimensions are incompatible.");
  }

  using V = tensor_scalar_t<Y>;
  const V alpha_v = static_cast<V>(alpha);

  __detail::assign_scaled_(beta, z, y);

  __backend::for_each(a, [&](auto&& e) {
    auto&& [idx, a_v] = e;
    auto&& [i, k] = idx;
    for (tensor_index_t<Y> j = 0; j < __backend::shape(x)[1]; j++) {
      __backend::lookup(y, i, j) +=
          alpha_v * (a_v * __backend::lookup(x, k, j));
    }
  });
}

// SpMV / SpMM: y = alpha * A * x
template <__detail::scalar Alpha, matrix A, tensor X, tensor Y>
  requires(__backend::lookupable<X> && __backend::lookupable<Y>)
void multiply(operation_state& state, Alpha alpha, A&& a, X&& x, Y&& y) {
  log_trace("");
  multiply(state, alpha, std::forward<A>(a), std::forward<X>(x),
           tensor_scalar_t<Y>(0), y, y);
}

// SpMV / SpMM / SpGEMM inspect. The reference implementation has nothing to
// prepare.
template <__detail::scalar Alpha, matrix A, tensor X, tensor Y>
void multiply_inspect(operation_state& state, Alpha alpha, A&& a, X&& x,
                      Y&& y) {
  log_trace("");
}

template <__detail::scalar Alpha, matrix A, tensor X, __detail::scalar Beta,
          tensor Z, tensor Y>
void multiply_inspect(operation_state& state, Alpha alpha, A&& a, X&& x,
                      Beta beta, Z&& z, Y&& y) {
  log_trace("");
}

// SpGEMM compute stage: C = alpha * A * B. Sets state.get_result_nnz().
template <__detail::scalar Alpha, matrix A, matrix B, matrix C>
  requires((__detail::is_csr_wrapper_v<C> || __detail::is_csc_wrapper_v<C>) &&
           requires(A&& a, B&& b, C&& c) { multiply_compute(a, b, c); })
void multiply_compute(operation_state& state, Alpha alpha, A&& a, B&& b,
                      C&& c) {
  log_trace("");
  auto info = multiply_compute(a, b, c);
  state.update_impl_(info.result_nnz());
}

// SpGEMM fill stage: C = alpha * A * B, into arrays sized by
// multiply_compute.
template <__detail::scalar Alpha, matrix A, matrix B, matrix C>
  requires((__detail::is_csr_wrapper_v<C> || __detail::is_csc_wrapper_v<C>) &&
           requires(A&& a, B&& b, C&& c) { multiply(a, b, c); })
void multiply_fill(operation_state& state, Alpha alpha, A&& a, B&& b, C&& c) {
  log_trace("");
  multiply(a, b, c);

  using V = tensor_scalar_t<C>;
  const V alpha_v = static_cast<V>(alpha);
  if (alpha_v != V(1)) {
    for (auto&& v : __backend::values(c)) {
      v *= alpha_v;
    }
  }
}

// Optional leading execution policy. The reference implementation runs
// sequentially and ignores it.
template <__detail::execution_policy P, typename... Args>
void multiply(P&& policy, operation_state& state, Args&&... args) {
  multiply(state, std::forward<Args>(args)...);
}

template <__detail::execution_policy P, typename... Args>
void multiply_inspect(P&& policy, operation_state& state, Args&&... args) {
  multiply_inspect(state, std::forward<Args>(args)...);
}

template <__detail::execution_policy P, typename... Args>
void multiply_compute(P&& policy, operation_state& state, Args&&... args) {
  multiply_compute(state, std::forward<Args>(args)...);
}

template <__detail::execution_policy P, typename... Args>
void multiply_fill(P&& policy, operation_state& state, Args&&... args) {
  multiply_fill(state, std::forward<Args>(args)...);
}

} // namespace spblas
