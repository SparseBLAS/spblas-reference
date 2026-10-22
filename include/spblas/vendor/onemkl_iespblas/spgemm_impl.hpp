#pragma once

#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>

#include <spblas/algorithms/detail/sparse_output.hpp>
#include <spblas/concepts.hpp>
#include <spblas/detail/api_concepts.hpp>
#include <spblas/detail/log.hpp>
#include <spblas/detail/operation_state.hpp>

#include "detail/exported.hpp"
#include "detail/operand.hpp"
#include "mkl_wrappers.hpp"
#include "operation_state_t.hpp"

//
// oneMKL IE SpGEMM: C = alpha * A * B, via the staged mkl_sparse_sp2m
//
//   stage                      sp2m request          then
//   multiply_compute           NNZ_COUNT             set get_result_nnz()
//   multiply_fill              FINALIZE_MULT         copy C (pattern, alpha *
//                                                    values) into the user's
//                                                    arrays; release the state
//   multiply_compute_symbolic  NNZ_COUNT             set get_result_nnz()
//   multiply_fill_symbolic     FINALIZE_MULT_NO_VAL  copy C's pattern
//   multiply_numeric           FINALIZE_MULT         copy alpha * values into
//                                                    the user's pattern
//   multiply_inspect does nothing: sp2m has no inspection step to prepare.
//
//   The state keeps the operand handles and C between stages, so the later
//   stages must be given the same A and B (checked); their values may change
//   between multiply_numeric calls, which then recompute only the values.
//   multiply_numeric right after multiply_compute_symbolic writes pattern and
//   values. When the user's pattern differs from C's (e.g. a superset), the
//   values are scattered into it, with the reference's checks.
//
//   A, B   as for SpMV (detail/operand.hpp); conj(A) without a transpose is
//          a conjugated copy of A's values held in the state (refreshed for
//          each FINALIZE stage). An operand sp2m transposes gets a fresh
//          handle for each FINALIZE stage, since sp2m keeps a transposed
//          copy of it inside the handle that would not see new values.
//          One index width for A, B, C.
//   C      CSR or CSC wrapper. For CSC, C^T = B^T * A^T is computed as CSR,
//          which gives C's CSC arrays.
//   MKL owns C's arrays, so filling copies them into the user's.
//
// Memory
//   A and B: handles wrap the user's arrays (no copy), so the arrays must stay
//     alive, with unchanged patterns, until the last stage. Exception: a
//     conjugated, non-transposed operand gets a conjugated copy of its values
//     (from the operation_state's allocator) for as long as its handle lives.
//   C: sp2m cannot write into user arrays. MKL allocates C's pattern and
//     values inside its handle (with MKL's allocator; the user's allocator
//     cannot redirect this), and the fill stages copy them into the user's
//     arrays. So C exists twice:
//       2-stage: only during multiply_fill, which releases MKL's copy.
//       3-stage: from multiply_fill_symbolic until the operation_state is
//         destroyed or reused by a compute stage, so that multiply_numeric
//         can rerun FINALIZE_MULT -- about nnz(C) * (sizeof(T) + sizeof(I))
//         extra bytes, plus the row pointers.
//     The operand handles (and any conjugated copies) are held as long.
//   multiply_numeric into a pattern that differs from MKL's also uses
//   nnz(C)-sized scratch, and one dense row (column, for CSC) of C, from the
//   state's allocator, for the duration of the call.
//

namespace spblas {

namespace __mkl_iespblas {

template <typename A, typename B, typename C>
concept spgemm_operands =
    sparse_input<A> && sparse_input<B> && __detail::compressed_output<C> &&
    sparse_input<C> && std::same_as<tensor_scalar_t<A>, tensor_scalar_t<C>> &&
    std::same_as<tensor_scalar_t<B>, tensor_scalar_t<C>> &&
    sizeof(tensor_index_t<A>) == sizeof(tensor_index_t<C>) &&
    sizeof(tensor_index_t<B>) == sizeof(tensor_index_t<C>);

template <typename A, typename B, typename C>
void check_spgemm_shapes(A&& a, B&& b, C&& c) {
  if (__backend::shape(a)[0] != __backend::shape(c)[0] ||
      __backend::shape(b)[1] != __backend::shape(c)[1] ||
      __backend::shape(a)[1] != __backend::shape(b)[0]) {
    throw std::invalid_argument(
        "multiply: matrix dimensions are incompatible.");
  }
}

// sp2m's operands, left * right: A, B for a CSR C; B^T, A^T for a CSC C.
//
// They never use a matrix_opt's cached handle: these handles are kept in the
// operation_state across calls, and a matrix_opt's handle can be dropped in
// between (update_matrix, scale). sp2m gains nothing from an optimized
// handle, and handles wrap the user's arrays either way, so new values are
// still seen by later stages.
template <typename C, typename A, typename B>
auto product_operands(A&& a, B&& b) {
  auto uncached = [](auto operand) {
    operand.opt = nullptr;
    return operand;
  };
  if constexpr (__detail::is_csr_wrapper_v<C>) {
    return std::pair(uncached(analyze(a)), uncached(analyze(b)));
  } else {
    return std::pair(uncached(transposed_operand(analyze(b))),
                     uncached(transposed_operand(analyze(a))));
  }
}

// The operand handles kept in the state between sp2m stages.
template <typename WL, typename WR>
struct product_handles {
  acquired_handle<WL> left;
  acquired_handle<WR> right;
};

template <typename C, typename A, typename B>
using product_handles_t = product_handles<
    decltype(product_operands<C>(std::declval<A>(), std::declval<B>())
                 .first.csr),
    decltype(product_operands<C>(std::declval<A>(), std::declval<B>())
                 .second.csr)>;

// Compute stages: acquire the operand handles, keep them in the state, and
// run sp2m(NNZ_COUNT) into the state's C. Returns C's nnz.
template <typename A, typename B, typename C>
std::size_t start_product(operation_state& state, A&& a, B&& b, C&& c) {
  check_spgemm_shapes(a, b, c);
  using P = product_handles_t<C, A, B>;
  using J = mkl_int_t<tensor_index_t<C>>;
  using T = tensor_scalar_t<C>;

  auto& s = state.state_;
  s.reset();

  auto allocator = state.get_allocator();
  auto [left, right] = product_operands<C>(a, b);
  auto handles = std::allocate_shared<P>(
      std::pmr::polymorphic_allocator<P>(allocator.resource()),
      P{{left, allocator, true}, {right, allocator, true}});
  s.operands = handles;
  s.operands_type = &typeid(P);

  auto& product = get_or_emplace<J>(s.c);
  sp2m(mkl_operation(handles->left.op()), general_descr(), handles->left.get(),
       mkl_operation(handles->right.op()), general_descr(),
       handles->right.get(), SPARSE_STAGE_NNZ_COUNT, product);

  return exported_nnz(export_csr<T, tensor_index_t<C>>(product));
}

// Later stages: check the state holds a product of these A and B, refresh
// conjugated copies, and run sp2m(request) on the state's C. Returns C.
template <typename A, typename B, typename C>
auto& finish_product(operation_state& state, A&& a, B&& b, C&& c,
                     sparse_request_t request, const char* operation) {
  check_spgemm_shapes(a, b, c);
  using P = product_handles_t<C, A, B>;
  using J = mkl_int_t<tensor_index_t<C>>;

  auto& s = state.state_;
  P* handles = s.template get_operands<P>();
  auto* product = std::get_if<sparse_handle<J>>(&s.c);
  if (handles == nullptr || product == nullptr || !*product) {
    throw std::invalid_argument(
        std::string(operation) +
        ": no product in this operation_state; run multiply_compute or "
        "multiply_compute_symbolic with it first.");
  }

  auto [left, right] = product_operands<C>(a, b);
  if (!handles->left.matches(left) || !handles->right.matches(right)) {
    throw std::invalid_argument(
        std::string(operation) +
        ": A and B must be the matrices given to the compute stage.");
  }

  handles->left.refresh();
  handles->right.refresh();
  sp2m(mkl_operation(handles->left.op()), general_descr(), handles->left.get(),
       mkl_operation(handles->right.op()), general_descr(),
       handles->right.get(), request, *product);
  return *product;
}

} // namespace __mkl_iespblas

template <__detail::scalar Alpha, matrix A, matrix B, matrix C>
  requires(__mkl_iespblas::spgemm_operands<A, B, C>)
void multiply_inspect(operation_state& state, Alpha alpha, A&& a, B&& b,
                      C&& c) {
  log_trace("");
}

template <__detail::scalar Alpha, matrix A, matrix B, matrix C>
  requires(__mkl_iespblas::spgemm_operands<A, B, C>)
void multiply_compute(operation_state& state, Alpha alpha, A&& a, B&& b,
                      C&& c) {
  log_trace("");
  state.update_impl_(__mkl_iespblas::start_product(state, a, b, c));
}

template <__detail::scalar Alpha, matrix A, matrix B, matrix C>
  requires(__mkl_iespblas::spgemm_operands<A, B, C>)
void multiply_fill(operation_state& state, Alpha alpha, A&& a, B&& b, C&& c) {
  log_trace("");
  using T = tensor_scalar_t<C>;
  using I = tensor_index_t<C>;
  auto& product = __mkl_iespblas::finish_product(
      state, a, b, c, SPARSE_STAGE_FINALIZE_MULT, "multiply_fill");
  __mkl_iespblas::copy_exported(__mkl_iespblas::export_csr<T, I>(product), c,
                                true, static_cast<T>(alpha), "multiply_fill");
  state.state_.reset();
}

template <__detail::scalar Alpha, matrix A, matrix B, matrix C>
  requires(__mkl_iespblas::spgemm_operands<A, B, C>)
void multiply_compute_symbolic(operation_state& state, Alpha alpha, A&& a,
                               B&& b, C&& c) {
  log_trace("");
  state.update_impl_(__mkl_iespblas::start_product(state, a, b, c));
}

template <__detail::scalar Alpha, matrix A, matrix B, matrix C>
  requires(__mkl_iespblas::spgemm_operands<A, B, C>)
void multiply_fill_symbolic(operation_state& state, Alpha alpha, A&& a, B&& b,
                            C&& c) {
  log_trace("");
  using T = tensor_scalar_t<C>;
  using I = tensor_index_t<C>;
  auto& product = __mkl_iespblas::finish_product(
      state, a, b, c, SPARSE_STAGE_FINALIZE_MULT_NO_VAL,
      "multiply_fill_symbolic");
  auto exported = __mkl_iespblas::export_csr<T, I>(product);
  __detail::fill_pattern_only(
      state, c, "multiply_fill_symbolic", [&](auto&& t) {
        __mkl_iespblas::copy_exported(exported, t, false, T(1),
                                      "multiply_fill_symbolic");
      });
  state.state_.pattern_filled = true;
}

template <__detail::scalar Alpha, matrix A, matrix B, matrix C>
  requires(__mkl_iespblas::spgemm_operands<A, B, C>)
void multiply_numeric(operation_state& state, Alpha alpha, A&& a, B&& b,
                      C&& c) {
  log_trace("");
  using T = tensor_scalar_t<C>;
  using I = tensor_index_t<C>;
  const T alpha_v = static_cast<T>(alpha);
  auto& product = __mkl_iespblas::finish_product(
      state, a, b, c, SPARSE_STAGE_FINALIZE_MULT, "multiply_numeric");
  auto exported = __mkl_iespblas::export_csr<T, I>(product);

  if (!state.state_.pattern_filled) {
    // No fill_symbolic stage: write the pattern too.
    __mkl_iespblas::copy_exported(exported, c, true, alpha_v,
                                  "multiply_numeric");
    state.state_.pattern_filled = true;
  } else if (__mkl_iespblas::same_pattern(exported, c)) {
    __mkl_iespblas::copy_exported_values(exported, c, alpha_v,
                                         "multiply_numeric");
  } else {
    __detail::fill_values_only(state, c, __mkl_iespblas::exported_nnz(exported),
                               "multiply_numeric", [&](auto&& t) {
                                 __mkl_iespblas::copy_exported(
                                     exported, t, true, alpha_v,
                                     "multiply_numeric");
                               });
  }
}

} // namespace spblas
