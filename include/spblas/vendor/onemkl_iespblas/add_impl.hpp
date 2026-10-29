#pragma once

#include <stdexcept>
#include <string>
#include <variant>

#include <spblas/algorithms/add_impl.hpp>
#include <spblas/algorithms/detail/spadd.hpp>
#include <spblas/algorithms/detail/sparse_output.hpp>
#include <spblas/concepts.hpp>
#include <spblas/detail/log.hpp>
#include <spblas/detail/operation_state.hpp>

#include "detail/exported.hpp"
#include "detail/operand.hpp"
#include "mkl_wrappers.hpp"
#include "operation_state_t.hpp"

//
// oneMKL IE SpADD: C = A + B via mkl_sparse_?_add (C = alpha * op(A) + B)
//
//   Used when the operands map onto it: B (for a CSC C: B^T) needs no
//   operation, and A is not conjugated without a transpose. That covers
//   all-CSR and all-CSC operands (and matrix_opt of them), and e.g.
//   view::transpose(A) + B, which the reference kernel cannot do. Other
//   combinations use the reference implementation (algorithms/add_impl.hpp).
//
//   stage                 does
//   add_compute           mkl_sparse_?_add into the state; set nnz
//   add_fill              copy the sum into C; release the state
//   add_compute_symbolic  as add_compute
//   add_fill_symbolic     copy the sum's pattern into C; release the state
//   add_numeric           mkl_sparse_?_add again (values may have changed),
//                         copy the values into C's pattern
//
//   mkl_sparse_?_add has no symbolic-only stage, so the sum is computed in
//   full by add_compute(_symbolic) and again by add_numeric. C is copied in
//   MKL's order, so the fill stages set C.sort_state() to `unknown`.
//
// Memory
//   As for SpGEMM (spgemm_impl.hpp), MKL allocates the sum inside its handle
//   and the fill stages copy it into the user's arrays. Here MKL's copy lives
//   in the operation_state only from add_compute(_symbolic) until
//   add_fill(_symbolic), which release it; add_numeric computes a new sum
//   and frees it before returning.
//

namespace spblas {

namespace __mkl_iespblas {

// Whether C = A + B (for a CSC C: C^T = A^T + B^T) maps onto
// alpha * op(A) + B.
template <typename A, typename B, typename C>
constexpr bool add_maps_to_mkl() {
  constexpr auto fa = operand_flags<A>();
  constexpr auto fb = operand_flags<B>();
  constexpr bool csc = __detail::is_csc_wrapper_v<C>;
  constexpr bool complex = __detail::is_std_complex_v<tensor_scalar_t<C>>;
  constexpr bool transpose_a = fa.transpose != csc;
  constexpr bool transpose_b = fb.transpose != csc;
  constexpr bool conjugate_a = fa.conjugate && complex;
  constexpr bool conjugate_b = fb.conjugate && complex;
  return !transpose_b && !conjugate_b && !(conjugate_a && !transpose_a);
}

template <typename A, typename B, typename C>
concept mkl_add_operands =
    sparse_input<A> && sparse_input<B> && __detail::compressed_output<C> &&
    sparse_input<C> && std::same_as<tensor_scalar_t<A>, tensor_scalar_t<C>> &&
    std::same_as<tensor_scalar_t<B>, tensor_scalar_t<C>> &&
    sizeof(tensor_index_t<A>) == sizeof(tensor_index_t<C>) &&
    sizeof(tensor_index_t<B>) == sizeof(tensor_index_t<C>) &&
    add_maps_to_mkl<A, B, C>();

// The operands the reference add kernel takes (same orientation).
template <typename A, typename B, typename C>
concept reference_add_operands =
    __detail::spadd_by_rows<A, B, C> || __detail::spadd_by_columns<A, B, C>;

// The sum (for a CSC C: its transpose) as a CSR handle in `out`.
template <typename A, typename B, typename C, typename J>
void sum(operation_state& state, A&& a, B&& b, C&& c, sparse_handle<J>& out) {
  if (__backend::shape(a) != __backend::shape(b) ||
      __backend::shape(b) != __backend::shape(c)) {
    throw std::invalid_argument("add: matrix dimensions are incompatible.");
  }
  using T = tensor_scalar_t<C>;
  auto allocator = state.get_allocator().host();
  auto operand_a = analyze(a);
  auto operand_b = analyze(b);
  if constexpr (__detail::is_csc_wrapper_v<C>) {
    operand_a = transposed_operand(operand_a);
    operand_b = transposed_operand(operand_b);
  }
  acquired_handle ha(operand_a, allocator, true);
  acquired_handle hb(operand_b, allocator, true);
  add(mkl_operation(ha.op()), ha.get(), T(1), hb.get(), out);
}

template <typename C>
using sum_handle_t = sparse_handle<mkl_int_t<tensor_index_t<C>>>;

template <typename C>
sum_handle_t<C>& computed_sum(operation_state& state, const char* operation) {
  auto* s = std::get_if<sum_handle_t<C>>(&state.state_.c);
  if (s == nullptr || !*s) {
    throw std::invalid_argument(
        std::string(operation) +
        ": no sum in this operation_state; run add_compute or "
        "add_compute_symbolic with it first.");
  }
  return *s;
}

template <typename A, typename B, typename C>
void add_compute_(operation_state& state, A&& a, B&& b, C&& c) {
  state.state_.reset();
  auto& s = get_or_emplace<mkl_int_t<tensor_index_t<C>>>(state.state_.c);
  sum(state, a, b, c, s);
  state.update_impl_(
      exported_nnz(export_csr<tensor_scalar_t<C>, tensor_index_t<C>>(s)));
}

template <typename A, typename B, typename C>
void add_fill_(operation_state& state, A&& a, B&& b, C&& c) {
  using T = tensor_scalar_t<C>;
  auto& s = computed_sum<C>(state, "add_fill");
  copy_exported(export_csr<T, tensor_index_t<C>>(s), c, true, T(1), "add_fill");
  __detail::set_output_sort_state(c, sort_state::unknown);
  state.state_.reset();
}

template <typename A, typename B, typename C>
void add_fill_symbolic_(operation_state& state, A&& a, B&& b, C&& c) {
  using T = tensor_scalar_t<C>;
  auto& s = computed_sum<C>(state, "add_fill_symbolic");
  auto exported = export_csr<T, tensor_index_t<C>>(s);
  __detail::fill_pattern_only(state, c, "add_fill_symbolic", [&](auto&& t) {
    copy_exported(exported, t, false, T(1), "add_fill_symbolic");
  });
  __detail::set_output_sort_state(c, sort_state::unknown);
  state.state_.reset();
}

template <typename A, typename B, typename C>
void add_numeric_(operation_state& state, A&& a, B&& b, C&& c) {
  using T = tensor_scalar_t<C>;
  sum_handle_t<C> s;
  sum(state, a, b, c, s);
  auto exported = export_csr<T, tensor_index_t<C>>(s);
  if (same_pattern(exported, c)) {
    copy_exported_values(exported, c, T(1), "add_numeric");
  } else {
    __detail::fill_values_only(
        state, c, exported_nnz(exported), "add_numeric", [&](auto&& t) {
          copy_exported(exported, t, true, T(1), "add_numeric");
        });
  }
}

} // namespace __mkl_iespblas

// Where the reference kernel also applies, these overloads are more
// constrained than the reference ones and take precedence; the second of
// each pair covers combinations only MKL handles.

template <matrix A, matrix B, matrix C>
  requires(__mkl_iespblas::reference_add_operands<A, B, C> &&
           __mkl_iespblas::mkl_add_operands<A, B, C>)
void add_compute(operation_state& state, A&& a, B&& b, C&& c) {
  log_trace("");
  __mkl_iespblas::add_compute_(state, a, b, c);
}

template <matrix A, matrix B, matrix C>
  requires(__mkl_iespblas::mkl_add_operands<A, B, C> &&
           !__mkl_iespblas::reference_add_operands<A, B, C>)
void add_compute(operation_state& state, A&& a, B&& b, C&& c) {
  log_trace("");
  __mkl_iespblas::add_compute_(state, a, b, c);
}

template <matrix A, matrix B, matrix C>
  requires(__mkl_iespblas::reference_add_operands<A, B, C> &&
           __mkl_iespblas::mkl_add_operands<A, B, C>)
void add_fill(operation_state& state, A&& a, B&& b, C&& c) {
  log_trace("");
  __mkl_iespblas::add_fill_(state, a, b, c);
}

template <matrix A, matrix B, matrix C>
  requires(__mkl_iespblas::mkl_add_operands<A, B, C> &&
           !__mkl_iespblas::reference_add_operands<A, B, C>)
void add_fill(operation_state& state, A&& a, B&& b, C&& c) {
  log_trace("");
  __mkl_iespblas::add_fill_(state, a, b, c);
}

template <matrix A, matrix B, matrix C>
  requires(__mkl_iespblas::reference_add_operands<A, B, C> &&
           __mkl_iespblas::mkl_add_operands<A, B, C>)
void add_compute_symbolic(operation_state& state, A&& a, B&& b, C&& c) {
  log_trace("");
  __mkl_iespblas::add_compute_(state, a, b, c);
}

template <matrix A, matrix B, matrix C>
  requires(__mkl_iespblas::mkl_add_operands<A, B, C> &&
           !__mkl_iespblas::reference_add_operands<A, B, C>)
void add_compute_symbolic(operation_state& state, A&& a, B&& b, C&& c) {
  log_trace("");
  __mkl_iespblas::add_compute_(state, a, b, c);
}

template <matrix A, matrix B, matrix C>
  requires(__mkl_iespblas::reference_add_operands<A, B, C> &&
           __mkl_iespblas::mkl_add_operands<A, B, C>)
void add_fill_symbolic(operation_state& state, A&& a, B&& b, C&& c) {
  log_trace("");
  __mkl_iespblas::add_fill_symbolic_(state, a, b, c);
}

template <matrix A, matrix B, matrix C>
  requires(__mkl_iespblas::mkl_add_operands<A, B, C> &&
           !__mkl_iespblas::reference_add_operands<A, B, C>)
void add_fill_symbolic(operation_state& state, A&& a, B&& b, C&& c) {
  log_trace("");
  __mkl_iespblas::add_fill_symbolic_(state, a, b, c);
}

template <matrix A, matrix B, matrix C>
  requires(__mkl_iespblas::reference_add_operands<A, B, C> &&
           __mkl_iespblas::mkl_add_operands<A, B, C>)
void add_numeric(operation_state& state, A&& a, B&& b, C&& c) {
  log_trace("");
  __mkl_iespblas::add_numeric_(state, a, b, c);
}

template <matrix A, matrix B, matrix C>
  requires(__mkl_iespblas::mkl_add_operands<A, B, C> &&
           !__mkl_iespblas::reference_add_operands<A, B, C>)
void add_numeric(operation_state& state, A&& a, B&& b, C&& c) {
  log_trace("");
  __mkl_iespblas::add_numeric_(state, a, b, c);
}

} // namespace spblas
