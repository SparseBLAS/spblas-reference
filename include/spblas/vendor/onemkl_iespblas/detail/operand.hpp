#pragma once

#include <memory_resource>
#include <type_traits>

#include <spblas/concepts.hpp>
#include <spblas/detail/mdspan.hpp>
#include <spblas/views/inspectors.hpp>
#include <spblas/views/transposed_view.hpp>
#include <spblas/views/view.hpp>

#include "../matrix_opt_state_t.hpp"
#include "../mkl_wrappers.hpp"

//
// How spblas operands map onto oneMKL:
//
//   sparse operand                 handle over         MKL operation
//   csr_wrapper A                  A                   NON_TRANSPOSE
//   csc_wrapper A                  A^T (same arrays)   TRANSPOSE
//   matrix_opt(A)                  as A, cached in the matrix_opt
//   view::transpose(X)             as X                X's op transposed
//   view::conjugate(X), complex    as X                X's op conjugated
//
// A conjugated, non-transposed operand has no MKL operation: mv / mm / trsv
// implement it in mkl_wrappers.hpp (no copy of A); sp2m and add use a
// conjugated copy of the values instead (see acquired_handle).
//
// Dense operands are mdspans with layout_right / layout_left (row / column
// major, default accessor), or view::transpose of one (the opposite layout,
// no copy). Other dense operands -- view::conjugate, or a layout that differs
// from the output's -- are materialized in the output's layout.
//

namespace spblas {

namespace __mkl_iespblas {

//
// Sparse operands
//

template <typename M>
constexpr bool is_sparse_input() {
  using U = std::remove_cvref_t<M>;
  if constexpr (__detail::is_csr_wrapper_v<U> ||
                __detail::is_csc_wrapper_v<U>) {
    return mkl_value<tensor_scalar_t<U>> && mkl_index<tensor_index_t<U>> &&
           std::same_as<tensor_index_t<U>, tensor_offset_t<U>>;
  } else if constexpr (__detail::is_matrix_opt_v<U> ||
                       __detail::is_conjugated_view_v<U>) {
    return is_sparse_input<decltype(std::declval<U>().base())>();
  } else if constexpr (__detail::is_transposed_view_v<U>) {
    return is_sparse_input<decltype(std::declval<U>().matrix())>();
  } else {
    return false;
  }
}

// A sparse operand oneMKL can take: CSR/CSC wrappers with an MKL value type
// and one 32- or 64-bit signed index type (MKL's CSR arrays share one integer
// type), possibly in matrix_opt / view::transpose / view::conjugate.
template <typename M>
concept sparse_input = is_sparse_input<M>();

// operand = op(handle over `csr`), where op transposes and/or conjugates.
template <typename W>
struct sparse_operand {
  W csr;
  bool transpose = false;
  bool conjugate = false;
  matrix_opt_state_t* opt = nullptr;
};

template <sparse_input M>
auto analyze(M&& m) {
  using U = std::remove_cvref_t<M>;
  if constexpr (__detail::is_csr_wrapper_v<U>) {
    return sparse_operand<U>{m};
  } else if constexpr (__detail::is_csc_wrapper_v<U>) {
    auto csr = spblas::view::transpose(m);
    return sparse_operand<decltype(csr)>{csr, true};
  } else if constexpr (__detail::is_matrix_opt_v<U>) {
    auto operand = analyze(m.base());
    operand.opt = m.state_.get();
    return operand;
  } else if constexpr (__detail::is_transposed_view_v<U>) {
    auto operand = analyze(m.matrix());
    operand.transpose = !operand.transpose;
    return operand;
  } else {
    auto operand = analyze(m.base());
    operand.conjugate = !operand.conjugate;
    return operand;
  }
}

// The operand with its transpose flag flipped: op(X)^T.
template <typename W>
sparse_operand<W> transposed_operand(sparse_operand<W> operand) {
  operand.transpose = !operand.transpose;
  return operand;
}

inline matrix_descr general_descr() {
  matrix_descr descr;
  descr.type = SPARSE_MATRIX_TYPE_GENERAL;
  descr.mode = SPARSE_FILL_MODE_FULL;
  descr.diag = SPARSE_DIAG_NON_UNIT;
  return descr;
}

// The transpose / conjugate flags analyze() gives an operand of type M,
// computed from the type alone.
struct operand_flags_t {
  bool transpose = false;
  bool conjugate = false;
};

template <sparse_input M>
constexpr operand_flags_t operand_flags() {
  using U = std::remove_cvref_t<M>;
  if constexpr (__detail::is_csr_wrapper_v<U>) {
    return {false, false};
  } else if constexpr (__detail::is_csc_wrapper_v<U>) {
    return {true, false};
  } else if constexpr (__detail::is_matrix_opt_v<U>) {
    return operand_flags<decltype(std::declval<U>().base())>();
  } else if constexpr (__detail::is_transposed_view_v<U>) {
    auto f = operand_flags<decltype(std::declval<U>().matrix())>();
    return {!f.transpose, f.conjugate};
  } else {
    auto f = operand_flags<decltype(std::declval<U>().base())>();
    return {f.transpose, !f.conjugate};
  }
}

// A handle for an operand, and the `operation` that applies it:
//   - the matrix_opt's cached handle when there is one, otherwise one owned
//     by this object;
//   - op() is `conjugate` for a conjugated, non-transposed operand, which
//     mv / mm / trsv carry out themselves. Callers of routines without that
//     (sp2m, add) pass materialize_conjugate, and get a handle over a
//     conjugated copy of the values (not cached) with NON_TRANSPOSE.
// The operand is remembered, so a handle kept across calls (e.g. between
// SpGEMM stages) can check it is used with the same matrix and refresh its
// conjugated copy after the values change.
template <typename W>
class acquired_handle {
public:
  using T = tensor_scalar_t<W>;
  using I = tensor_index_t<W>;
  using J = mkl_int_t<I>;

  acquired_handle(const sparse_operand<W>& operand,
                  std::pmr::polymorphic_allocator<> allocator,
                  bool materialize_conjugate = false)
      : operand_(operand), conjugated_values_(allocator) {
    const bool conjugate = operand.conjugate && __detail::is_std_complex_v<T>;

    op_ = !operand.transpose
              ? (conjugate ? operation::conjugate : operation::non_transpose)
          : conjugate ? operation::conjugate_transpose
                      : operation::transpose;

    if (op_ == operation::conjugate && materialize_conjugate) {
      materialized_ = true;
      refresh();
      op_ = operation::non_transpose;
      return;
    }

    if (operand.opt != nullptr) {
      cached_ = &get_or_emplace<J>(operand.opt->handle);
      if (!*cached_) {
        create(*cached_, operand.csr, operand.csr.values().data());
      }
    } else {
      create(local_, operand.csr, operand.csr.values().data());
    }
  }

  const sparse_handle<J>& get() const noexcept {
    return cached_ != nullptr ? *cached_ : local_;
  }

  operation op() const noexcept {
    return op_;
  }

  // Whether the handle lives in a matrix_opt (so optimizing it pays off).
  bool cached() const noexcept {
    return cached_ != nullptr;
  }

  // Whether `operand` is the matrix this handle was acquired for: the same
  // arrays, shape, flags and matrix_opt.
  bool matches(const sparse_operand<W>& operand) const noexcept {
    return operand.csr.rowptr().data() == operand_.csr.rowptr().data() &&
           operand.csr.colind().data() == operand_.csr.colind().data() &&
           operand.csr.values().data() == operand_.csr.values().data() &&
           operand.csr.shape() == operand_.csr.shape() &&
           operand.csr.size() == operand_.csr.size() &&
           operand.transpose == operand_.transpose &&
           operand.conjugate == operand_.conjugate &&
           operand.opt == operand_.opt;
  }

  // Makes the handle see the operand's current values, before reusing it
  // after they may have changed (e.g. between SpGEMM stages):
  //   - a conjugated copy re-reads them;
  //   - a handle of its own used with a (conjugate) transpose is created
  //     anew over the user's arrays: sp2m keeps a transposed copy inside
  //     such a handle, and a later FINALIZE_MULT would reuse its stale
  //     values (seen with oneMKL 2026.0);
  //   - other handles wrap the user's arrays and see new values directly.
  void refresh() {
    if (!materialized_) {
      if (cached_ == nullptr && (op_ == operation::transpose ||
                                 op_ == operation::conjugate_transpose)) {
        create(local_, operand_.csr, operand_.csr.values().data());
      }
      return;
    }
    // The copy keeps its size (the matrix's nnz), so its buffer -- which the
    // handle wraps -- does not move after the first call.
    auto values = operand_.csr.values();
    conjugated_values_.resize(values.size());
    for (std::size_t i = 0; i < values.size(); i++) {
      conjugated_values_[i] = conj_if_complex(values[i]);
    }
    if (!local_) {
      create(local_, operand_.csr, conjugated_values_.data());
    }
  }

private:
  static void create(sparse_handle<J>& h, const W& csr, T* values) {
    auto shape = csr.shape();
    auto rowptr = csr.rowptr().data();
    create_csr<T, I>(h, SPARSE_INDEX_BASE_ZERO, shape[0], shape[1], rowptr,
                     rowptr + 1, csr.colind().data(), values);
  }

  sparse_operand<W> operand_;
  sparse_handle<J> local_;
  sparse_handle<J>* cached_ = nullptr;
  std::pmr::vector<T> conjugated_values_;
  operation op_;
  bool materialized_ = false;
};

//
// Dense operands
//

template <typename M>
constexpr bool is_plain_mdspan() {
  using U = std::remove_cvref_t<M>;
  if constexpr (__detail::is_matrix_mdspan_v<U>) {
    using layout = typename U::layout_type;
    return mkl_value<typename U::value_type> &&
           std::is_same_v<typename U::accessor_type,
                          __mdspan::default_accessor<typename U::value_type>> &&
           (std::is_same_v<layout, __mdspan::layout_right> ||
            std::is_same_v<layout, __mdspan::layout_left>);
  } else {
    return false;
  }
}

// A dense matrix MKL can use in place: a row- or column-major mdspan, or
// view::transpose of one.
template <typename M>
concept direct_dense =
    is_plain_mdspan<M>() ||
    (__detail::is_transposed_view_v<M> &&
     is_plain_mdspan<decltype(std::declval<M>().matrix())>());

// Pointer, layout and leading dimension of a direct dense matrix.
template <typename T, typename J>
struct dense_operand {
  T* data;
  sparse_layout_t layout;
  J ld;
};

template <typename J, typename M>
auto describe_dense(M&& m) {
  using U = std::remove_cvref_t<M>;
  if constexpr (__detail::is_transposed_view_v<U>) {
    auto d = describe_dense<J>(m.matrix());
    d.layout = d.layout == SPARSE_LAYOUT_ROW_MAJOR ? SPARSE_LAYOUT_COLUMN_MAJOR
                                                   : SPARSE_LAYOUT_ROW_MAJOR;
    return d;
  } else {
    using T = typename U::value_type;
    if constexpr (std::is_same_v<typename U::layout_type,
                                 __mdspan::layout_right>) {
      return dense_operand<T, J>{m.data_handle(), SPARSE_LAYOUT_ROW_MAJOR,
                                 J(m.stride(0))};
    } else {
      return dense_operand<T, J>{m.data_handle(), SPARSE_LAYOUT_COLUMN_MAJOR,
                                 J(m.stride(1))};
    }
  }
}

// `x` as a dense operand in `layout`: in place if possible, otherwise copied
// (through element lookup, so views are applied) into `scratch`.
template <typename J, typename T, matrix X>
dense_operand<const T, J> dense_input(X&& x, sparse_layout_t layout,
                                      std::pmr::vector<T>& scratch) {
  if constexpr (direct_dense<X>) {
    auto d = describe_dense<J>(x);
    if (d.layout == layout) {
      return {d.data, d.layout, d.ld};
    }
  }

  const std::size_t rows = __backend::shape(x)[0];
  const std::size_t cols = __backend::shape(x)[1];
  scratch.resize(rows * cols);
  const bool row_major = layout == SPARSE_LAYOUT_ROW_MAJOR;
  for (std::size_t i = 0; i < rows; i++) {
    for (std::size_t j = 0; j < cols; j++) {
      scratch[row_major ? i * cols + j : j * rows + i] =
          __backend::lookup(x, tensor_index_t<X>(i), tensor_index_t<X>(j));
    }
  }
  return {scratch.data(), layout, J(row_major ? cols : rows)};
}

// A dense vector's contiguous data, or a copy (applying e.g. conjugation)
// in `scratch`.
template <typename T, vector X>
const T* vector_input(X&& x, std::pmr::vector<T>& scratch) {
  if constexpr (__ranges::contiguous_range<X>) {
    return __ranges::data(x);
  } else {
    const std::size_t n = __backend::shape(x);
    scratch.resize(n);
    for (std::size_t i = 0; i < n; i++) {
      scratch[i] = __backend::lookup(x, tensor_index_t<X>(i));
    }
    return scratch.data();
  }
}

} // namespace __mkl_iespblas

} // namespace spblas
