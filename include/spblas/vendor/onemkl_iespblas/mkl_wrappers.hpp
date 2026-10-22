#pragma once

#include <complex>
#include <concepts>
#include <cstring>
#include <memory_resource>
#include <new>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "mkl.h"

//
// C++ wrappers over the oneMKL Inspector-Executor Sparse BLAS C API. This is
// the only file that names MKL entry points; operation code calls these
// templates and never branches on value type or index width.
//
//   - Value types: float, double, std::complex<float>, std::complex<double>
//     (mkl_sparse_{s,d,c,z}_*).
//   - Index types: any 32- or 64-bit signed integer, classified by size (so
//     std::int64_t, which is `long`, works although MKL_INT64 is `long
//     long`). 32-bit indices use the plain entry points, 64-bit ones the
//     *_64 variants; the backend requires the LP64 interface.
//   - Handles are owned by sparse_handle<J>, whose J (MKL_INT or MKL_INT64)
//     records which family created them, so the families cannot be mixed.
//   - Every sparse_status_t is checked; failures throw.
//   - `operation` extends sparse_operation_t with `conjugate` (conj(A), not
//     transposed), which MKL lacks. mv / mm / trsv implement it here via
//     conj(A) x = conj(A conj(x)), without copying A; other routines take
//     sparse_operation_t, and their callers materialize conj(A) instead.
//
// Unsupported value or index types fail at compile time.
//

namespace spblas {

namespace __mkl_iespblas {

template <typename I>
concept mkl_index =
    std::signed_integral<I> && (sizeof(I) == 4 || sizeof(I) == 8);

template <typename I>
inline constexpr bool is_64 = sizeof(I) == 8;

// The MKL integer type with I's width.
template <typename I>
using mkl_int_t = std::conditional_t<is_64<I>, MKL_INT64, MKL_INT>;

template <typename T>
concept mkl_value = std::same_as<T, float> || std::same_as<T, double> ||
                    std::same_as<T, std::complex<float>> ||
                    std::same_as<T, std::complex<double>>;

// The MKL value type with T's layout.
template <typename T>
using mkl_value_t =
    std::conditional_t<std::same_as<T, std::complex<float>>, MKL_Complex8,
                       std::conditional_t<std::same_as<T, std::complex<double>>,
                                          MKL_Complex16, T>>;

template <mkl_index I>
mkl_int_t<I>* to_mkl(I* p) noexcept {
  return reinterpret_cast<mkl_int_t<I>*>(p);
}

template <mkl_value T>
mkl_value_t<T>* to_mkl(T* p) noexcept {
  return reinterpret_cast<mkl_value_t<T>*>(p);
}

template <mkl_value T>
const mkl_value_t<T>* to_mkl(const T* p) noexcept {
  return reinterpret_cast<const mkl_value_t<T>*>(p);
}

template <mkl_value T>
mkl_value_t<T> to_mkl_scalar(T v) noexcept {
  mkl_value_t<T> r;
  static_assert(sizeof(r) == sizeof(v));
  std::memcpy(&r, &v, sizeof(r));
  return r;
}

inline const char* status_name(sparse_status_t status) noexcept {
  switch (status) {
  case SPARSE_STATUS_SUCCESS:
    return "SPARSE_STATUS_SUCCESS";
  case SPARSE_STATUS_NOT_INITIALIZED:
    return "SPARSE_STATUS_NOT_INITIALIZED";
  case SPARSE_STATUS_ALLOC_FAILED:
    return "SPARSE_STATUS_ALLOC_FAILED";
  case SPARSE_STATUS_INVALID_VALUE:
    return "SPARSE_STATUS_INVALID_VALUE";
  case SPARSE_STATUS_EXECUTION_FAILED:
    return "SPARSE_STATUS_EXECUTION_FAILED";
  case SPARSE_STATUS_INTERNAL_ERROR:
    return "SPARSE_STATUS_INTERNAL_ERROR";
  case SPARSE_STATUS_NOT_SUPPORTED:
    return "SPARSE_STATUS_NOT_SUPPORTED";
  }
  return "unknown sparse_status_t";
}

inline void throw_if_error(sparse_status_t status, const char* routine) {
  if (status == SPARSE_STATUS_SUCCESS) {
    return;
  }
  if (status == SPARSE_STATUS_ALLOC_FAILED) {
    throw std::bad_alloc();
  }
  std::string message =
      std::string("oneMKL ") + routine + " failed: " + status_name(status);
  if (status == SPARSE_STATUS_INVALID_VALUE) {
    throw std::invalid_argument(message);
  }
  throw std::runtime_error(message);
}

// MKL's sparse operations plus `conjugate`.
enum class operation {
  non_transpose,
  transpose,
  conjugate_transpose,
  conjugate
};

inline bool transposes(operation op) noexcept {
  return op == operation::transpose || op == operation::conjugate_transpose;
}

// The MKL operation that is passed to MKL for `op` (`conjugate` is carried
// out around a NON_TRANSPOSE call).
inline sparse_operation_t mkl_operation(operation op) noexcept {
  switch (op) {
  case operation::transpose:
    return SPARSE_OPERATION_TRANSPOSE;
  case operation::conjugate_transpose:
    return SPARSE_OPERATION_CONJUGATE_TRANSPOSE;
  default:
    return SPARSE_OPERATION_NON_TRANSPOSE;
  }
}

template <mkl_value T>
T conj_if_complex(T v) noexcept {
  if constexpr (std::same_as<T, std::complex<float>> ||
                std::same_as<T, std::complex<double>>) {
    return std::conj(v);
  } else {
    return v;
  }
}

// Owns a sparse_matrix_t created by the plain (J = MKL_INT) or the _64
// (J = MKL_INT64) family of entry points.
template <typename J>
  requires(std::same_as<J, MKL_INT> || std::same_as<J, MKL_INT64>)
class sparse_handle {
public:
  using index_type = J;

  sparse_handle() = default;
  sparse_handle(const sparse_handle&) = delete;
  sparse_handle& operator=(const sparse_handle&) = delete;

  sparse_handle(sparse_handle&& other) noexcept
      : handle_(std::exchange(other.handle_, nullptr)) {}

  sparse_handle& operator=(sparse_handle&& other) noexcept {
    if (this != &other) {
      reset();
      handle_ = std::exchange(other.handle_, nullptr);
    }
    return *this;
  }

  ~sparse_handle() {
    reset();
  }

  void reset() noexcept {
    if (handle_ != nullptr) {
      if constexpr (is_64<J>) {
        mkl_sparse_destroy_64(handle_);
      } else {
        mkl_sparse_destroy(handle_);
      }
      handle_ = nullptr;
    }
  }

  sparse_matrix_t get() const noexcept {
    return handle_;
  }

  // For routines that create a handle: releases the current one first.
  sparse_matrix_t* replace() noexcept {
    reset();
    return &handle_;
  }

  // For routines that update an existing handle in place.
  sparse_matrix_t* in_place() noexcept {
    return &handle_;
  }

  explicit operator bool() const noexcept {
    return handle_ != nullptr;
  }

private:
  sparse_matrix_t handle_ = nullptr;
};

template <typename I>
using handle_for_t = sparse_handle<mkl_int_t<I>>;

//
// mkl_sparse_?_create_csr: wraps user arrays (no copy).
//
template <mkl_value T, mkl_index I>
void create_csr(handle_for_t<I>& a, sparse_index_base_t base, I rows, I cols,
                I* rows_start, I* rows_end, I* col_indx, T* values) {
  sparse_matrix_t* out = a.replace();
  sparse_status_t s;
  if constexpr (!is_64<I>) {
    if constexpr (std::same_as<T, float>) {
      s = mkl_sparse_s_create_csr(out, base, rows, cols, to_mkl(rows_start),
                                  to_mkl(rows_end), to_mkl(col_indx), values);
    } else if constexpr (std::same_as<T, double>) {
      s = mkl_sparse_d_create_csr(out, base, rows, cols, to_mkl(rows_start),
                                  to_mkl(rows_end), to_mkl(col_indx), values);
    } else if constexpr (std::same_as<T, std::complex<float>>) {
      s = mkl_sparse_c_create_csr(out, base, rows, cols, to_mkl(rows_start),
                                  to_mkl(rows_end), to_mkl(col_indx),
                                  to_mkl(values));
    } else {
      s = mkl_sparse_z_create_csr(out, base, rows, cols, to_mkl(rows_start),
                                  to_mkl(rows_end), to_mkl(col_indx),
                                  to_mkl(values));
    }
  } else {
    if constexpr (std::same_as<T, float>) {
      s = mkl_sparse_s_create_csr_64(out, base, rows, cols, to_mkl(rows_start),
                                     to_mkl(rows_end), to_mkl(col_indx),
                                     values);
    } else if constexpr (std::same_as<T, double>) {
      s = mkl_sparse_d_create_csr_64(out, base, rows, cols, to_mkl(rows_start),
                                     to_mkl(rows_end), to_mkl(col_indx),
                                     values);
    } else if constexpr (std::same_as<T, std::complex<float>>) {
      s = mkl_sparse_c_create_csr_64(out, base, rows, cols, to_mkl(rows_start),
                                     to_mkl(rows_end), to_mkl(col_indx),
                                     to_mkl(values));
    } else {
      s = mkl_sparse_z_create_csr_64(out, base, rows, cols, to_mkl(rows_start),
                                     to_mkl(rows_end), to_mkl(col_indx),
                                     to_mkl(values));
    }
  }
  throw_if_error(s, "mkl_sparse_?_create_csr");
}

//
// mkl_sparse_?_export_csr: pointers into MKL-owned (or wrapped) arrays.
//
template <typename T, typename I>
struct exported_csr {
  sparse_index_base_t base;
  I rows;
  I cols;
  I* rows_start;
  I* rows_end;
  I* col_indx;
  T* values; // may be null for a pattern-only handle
};

template <mkl_value T, mkl_index I>
exported_csr<T, I> export_csr(const handle_for_t<I>& a) {
  using J = mkl_int_t<I>;
  sparse_index_base_t base;
  J rows, cols;
  J *rows_start, *rows_end, *col_indx;
  mkl_value_t<T>* values = nullptr;
  sparse_status_t s;
  if constexpr (!is_64<I>) {
    if constexpr (std::same_as<T, float>) {
      s = mkl_sparse_s_export_csr(a.get(), &base, &rows, &cols, &rows_start,
                                  &rows_end, &col_indx, &values);
    } else if constexpr (std::same_as<T, double>) {
      s = mkl_sparse_d_export_csr(a.get(), &base, &rows, &cols, &rows_start,
                                  &rows_end, &col_indx, &values);
    } else if constexpr (std::same_as<T, std::complex<float>>) {
      s = mkl_sparse_c_export_csr(a.get(), &base, &rows, &cols, &rows_start,
                                  &rows_end, &col_indx, &values);
    } else {
      s = mkl_sparse_z_export_csr(a.get(), &base, &rows, &cols, &rows_start,
                                  &rows_end, &col_indx, &values);
    }
  } else {
    if constexpr (std::same_as<T, float>) {
      s = mkl_sparse_s_export_csr_64(a.get(), &base, &rows, &cols, &rows_start,
                                     &rows_end, &col_indx, &values);
    } else if constexpr (std::same_as<T, double>) {
      s = mkl_sparse_d_export_csr_64(a.get(), &base, &rows, &cols, &rows_start,
                                     &rows_end, &col_indx, &values);
    } else if constexpr (std::same_as<T, std::complex<float>>) {
      s = mkl_sparse_c_export_csr_64(a.get(), &base, &rows, &cols, &rows_start,
                                     &rows_end, &col_indx, &values);
    } else {
      s = mkl_sparse_z_export_csr_64(a.get(), &base, &rows, &cols, &rows_start,
                                     &rows_end, &col_indx, &values);
    }
  }
  throw_if_error(s, "mkl_sparse_?_export_csr");
  return {base,
          I(rows),
          I(cols),
          reinterpret_cast<I*>(rows_start),
          reinterpret_cast<I*>(rows_end),
          reinterpret_cast<I*>(col_indx),
          reinterpret_cast<T*>(values)};
}

//
// mkl_sparse_?_mv: y = alpha * op(A) * x + beta * y
//
template <mkl_value T, typename J>
void mv(sparse_operation_t op, T alpha, const sparse_handle<J>& a,
        matrix_descr descr, const T* x, T beta, T* y) {
  sparse_status_t s;
  if constexpr (!is_64<J>) {
    if constexpr (std::same_as<T, float>) {
      s = mkl_sparse_s_mv(op, alpha, a.get(), descr, x, beta, y);
    } else if constexpr (std::same_as<T, double>) {
      s = mkl_sparse_d_mv(op, alpha, a.get(), descr, x, beta, y);
    } else if constexpr (std::same_as<T, std::complex<float>>) {
      s = mkl_sparse_c_mv(op, to_mkl_scalar(alpha), a.get(), descr, to_mkl(x),
                          to_mkl_scalar(beta), to_mkl(y));
    } else {
      s = mkl_sparse_z_mv(op, to_mkl_scalar(alpha), a.get(), descr, to_mkl(x),
                          to_mkl_scalar(beta), to_mkl(y));
    }
  } else {
    if constexpr (std::same_as<T, float>) {
      s = mkl_sparse_s_mv_64(op, alpha, a.get(), descr, x, beta, y);
    } else if constexpr (std::same_as<T, double>) {
      s = mkl_sparse_d_mv_64(op, alpha, a.get(), descr, x, beta, y);
    } else if constexpr (std::same_as<T, std::complex<float>>) {
      s = mkl_sparse_c_mv_64(op, to_mkl_scalar(alpha), a.get(), descr,
                             to_mkl(x), to_mkl_scalar(beta), to_mkl(y));
    } else {
      s = mkl_sparse_z_mv_64(op, to_mkl_scalar(alpha), a.get(), descr,
                             to_mkl(x), to_mkl_scalar(beta), to_mkl(y));
    }
  }
  throw_if_error(s, "mkl_sparse_?_mv");
}

//
// mkl_sparse_?_mm: Y = alpha * op(A) * X + beta * Y
//
template <mkl_value T, typename J>
void mm(sparse_operation_t op, T alpha, const sparse_handle<J>& a,
        matrix_descr descr, sparse_layout_t layout, const T* x, J columns,
        J ldx, T beta, T* y, J ldy) {
  sparse_status_t s;
  if constexpr (!is_64<J>) {
    if constexpr (std::same_as<T, float>) {
      s = mkl_sparse_s_mm(op, alpha, a.get(), descr, layout, x, columns, ldx,
                          beta, y, ldy);
    } else if constexpr (std::same_as<T, double>) {
      s = mkl_sparse_d_mm(op, alpha, a.get(), descr, layout, x, columns, ldx,
                          beta, y, ldy);
    } else if constexpr (std::same_as<T, std::complex<float>>) {
      s = mkl_sparse_c_mm(op, to_mkl_scalar(alpha), a.get(), descr, layout,
                          to_mkl(x), columns, ldx, to_mkl_scalar(beta),
                          to_mkl(y), ldy);
    } else {
      s = mkl_sparse_z_mm(op, to_mkl_scalar(alpha), a.get(), descr, layout,
                          to_mkl(x), columns, ldx, to_mkl_scalar(beta),
                          to_mkl(y), ldy);
    }
  } else {
    if constexpr (std::same_as<T, float>) {
      s = mkl_sparse_s_mm_64(op, alpha, a.get(), descr, layout, x, columns, ldx,
                             beta, y, ldy);
    } else if constexpr (std::same_as<T, double>) {
      s = mkl_sparse_d_mm_64(op, alpha, a.get(), descr, layout, x, columns, ldx,
                             beta, y, ldy);
    } else if constexpr (std::same_as<T, std::complex<float>>) {
      s = mkl_sparse_c_mm_64(op, to_mkl_scalar(alpha), a.get(), descr, layout,
                             to_mkl(x), columns, ldx, to_mkl_scalar(beta),
                             to_mkl(y), ldy);
    } else {
      s = mkl_sparse_z_mm_64(op, to_mkl_scalar(alpha), a.get(), descr, layout,
                             to_mkl(x), columns, ldx, to_mkl_scalar(beta),
                             to_mkl(y), ldy);
    }
  }
  throw_if_error(s, "mkl_sparse_?_mm");
}

//
// mkl_sparse_?_trsv: solve op(A) * y = alpha * x for y
//
template <mkl_value T, typename J>
void trsv(sparse_operation_t op, T alpha, const sparse_handle<J>& a,
          matrix_descr descr, const T* x, T* y) {
  sparse_status_t s;
  if constexpr (!is_64<J>) {
    if constexpr (std::same_as<T, float>) {
      s = mkl_sparse_s_trsv(op, alpha, a.get(), descr, x, y);
    } else if constexpr (std::same_as<T, double>) {
      s = mkl_sparse_d_trsv(op, alpha, a.get(), descr, x, y);
    } else if constexpr (std::same_as<T, std::complex<float>>) {
      s = mkl_sparse_c_trsv(op, to_mkl_scalar(alpha), a.get(), descr, to_mkl(x),
                            to_mkl(y));
    } else {
      s = mkl_sparse_z_trsv(op, to_mkl_scalar(alpha), a.get(), descr, to_mkl(x),
                            to_mkl(y));
    }
  } else {
    if constexpr (std::same_as<T, float>) {
      s = mkl_sparse_s_trsv_64(op, alpha, a.get(), descr, x, y);
    } else if constexpr (std::same_as<T, double>) {
      s = mkl_sparse_d_trsv_64(op, alpha, a.get(), descr, x, y);
    } else if constexpr (std::same_as<T, std::complex<float>>) {
      s = mkl_sparse_c_trsv_64(op, to_mkl_scalar(alpha), a.get(), descr,
                               to_mkl(x), to_mkl(y));
    } else {
      s = mkl_sparse_z_trsv_64(op, to_mkl_scalar(alpha), a.get(), descr,
                               to_mkl(x), to_mkl(y));
    }
  }
  throw_if_error(s, "mkl_sparse_?_trsv");
}

//
// mv / mm / trsv for any `operation`, including `conjugate`. Sizes are those
// of op(A) * x = y (mv, trsv: x has `x_size`, y has `y_size` entries; mm: X
// and Y have `x_rows` / `y_rows` rows and `columns` columns in `layout`).
// Temporaries come from `allocator`.
//

template <mkl_value T>
void conjugate_in_place(T* data, std::size_t n) {
  for (std::size_t i = 0; i < n; i++) {
    data[i] = conj_if_complex(data[i]);
  }
}

// Conjugates the rows x columns matrix at `data` (leading dimension ld).
template <mkl_value T, typename J>
void conjugate_in_place(T* data, sparse_layout_t layout, J rows, J columns,
                        J ld) {
  const bool row_major = layout == SPARSE_LAYOUT_ROW_MAJOR;
  const J outer = row_major ? rows : columns;
  const J inner = row_major ? columns : rows;
  for (J o = 0; o < outer; o++) {
    conjugate_in_place(data + o * ld, std::size_t(inner));
  }
}

template <mkl_value T, typename J>
void mv(operation op, T alpha, const sparse_handle<J>& a, matrix_descr descr,
        const T* x, J x_size, T beta, T* y, J y_size,
        std::pmr::polymorphic_allocator<> allocator) {
  if (op != operation::conjugate) {
    mv(mkl_operation(op), alpha, a, descr, x, beta, y);
    return;
  }
  // y = alpha conj(A) x + beta y  <=>  conj(y) = conj(alpha) A conj(x) +
  // conj(beta) conj(y)
  std::pmr::vector<T> x_conj(x, x + x_size, allocator);
  conjugate_in_place(x_conj.data(), x_conj.size());
  conjugate_in_place(y, std::size_t(y_size));
  mv(SPARSE_OPERATION_NON_TRANSPOSE, conj_if_complex(alpha), a, descr,
     x_conj.data(), conj_if_complex(beta), y);
  conjugate_in_place(y, std::size_t(y_size));
}

template <mkl_value T, typename J>
void mm(operation op, T alpha, const sparse_handle<J>& a, matrix_descr descr,
        sparse_layout_t layout, const T* x, J x_rows, J columns, J ldx, T beta,
        T* y, J y_rows, J ldy, std::pmr::polymorphic_allocator<> allocator) {
  if (op != operation::conjugate) {
    mm(mkl_operation(op), alpha, a, descr, layout, x, columns, ldx, beta, y,
       ldy);
    return;
  }
  // X's extent: (outer - 1) * ldx + inner elements.
  const bool row_major = layout == SPARSE_LAYOUT_ROW_MAJOR;
  const std::size_t outer = row_major ? x_rows : columns;
  const std::size_t inner = row_major ? columns : x_rows;
  const std::size_t extent = outer == 0 ? 0 : (outer - 1) * ldx + inner;
  std::pmr::vector<T> x_conj(x, x + extent, allocator);
  conjugate_in_place(x_conj.data(), layout, x_rows, columns, ldx);
  conjugate_in_place(y, layout, y_rows, columns, ldy);
  mm(SPARSE_OPERATION_NON_TRANSPOSE, conj_if_complex(alpha), a, descr, layout,
     x_conj.data(), columns, ldx, conj_if_complex(beta), y, ldy);
  conjugate_in_place(y, layout, y_rows, columns, ldy);
}

template <mkl_value T, typename J>
void trsv(operation op, T alpha, const sparse_handle<J>& a, matrix_descr descr,
          const T* x, T* y, J size,
          std::pmr::polymorphic_allocator<> allocator) {
  if (op != operation::conjugate) {
    trsv(mkl_operation(op), alpha, a, descr, x, y);
    return;
  }
  // conj(A) y = alpha x  <=>  A conj(y) = conj(alpha) conj(x)
  std::pmr::vector<T> x_conj(x, x + size, allocator);
  conjugate_in_place(x_conj.data(), x_conj.size());
  trsv(SPARSE_OPERATION_NON_TRANSPOSE, conj_if_complex(alpha), a, descr,
       x_conj.data(), y);
  conjugate_in_place(y, std::size_t(size));
}

//
// mkl_sparse_?_add: C = alpha * op(A) + B (creates C)
//
template <mkl_value T, typename J>
void add(sparse_operation_t op, const sparse_handle<J>& a, T alpha,
         const sparse_handle<J>& b, sparse_handle<J>& c) {
  sparse_matrix_t* out = c.replace();
  sparse_status_t s;
  if constexpr (!is_64<J>) {
    if constexpr (std::same_as<T, float>) {
      s = mkl_sparse_s_add(op, a.get(), alpha, b.get(), out);
    } else if constexpr (std::same_as<T, double>) {
      s = mkl_sparse_d_add(op, a.get(), alpha, b.get(), out);
    } else if constexpr (std::same_as<T, std::complex<float>>) {
      s = mkl_sparse_c_add(op, a.get(), to_mkl_scalar(alpha), b.get(), out);
    } else {
      s = mkl_sparse_z_add(op, a.get(), to_mkl_scalar(alpha), b.get(), out);
    }
  } else {
    if constexpr (std::same_as<T, float>) {
      s = mkl_sparse_s_add_64(op, a.get(), alpha, b.get(), out);
    } else if constexpr (std::same_as<T, double>) {
      s = mkl_sparse_d_add_64(op, a.get(), alpha, b.get(), out);
    } else if constexpr (std::same_as<T, std::complex<float>>) {
      s = mkl_sparse_c_add_64(op, a.get(), to_mkl_scalar(alpha), b.get(), out);
    } else {
      s = mkl_sparse_z_add_64(op, a.get(), to_mkl_scalar(alpha), b.get(), out);
    }
  }
  throw_if_error(s, "mkl_sparse_?_add");
}

//
// Type-generic routines (the handle's width selects the family)
//

// mkl_sparse_sp2m: C = op(A) * op(B), staged by `request`. For stages that
// create C, C's previous handle is released; for finalize stages pass the C
// produced by the matching earlier stage.
template <typename J>
void sp2m(sparse_operation_t op_a, matrix_descr descr_a,
          const sparse_handle<J>& a, sparse_operation_t op_b,
          matrix_descr descr_b, const sparse_handle<J>& b,
          sparse_request_t request, sparse_handle<J>& c) {
  const bool creates_c = request == SPARSE_STAGE_FULL_MULT ||
                         request == SPARSE_STAGE_NNZ_COUNT ||
                         request == SPARSE_STAGE_FULL_MULT_NO_VAL;
  sparse_matrix_t* out = creates_c ? c.replace() : c.in_place();
  sparse_status_t s;
  if constexpr (!is_64<J>) {
    s = mkl_sparse_sp2m(op_a, descr_a, a.get(), op_b, descr_b, b.get(), request,
                        out);
  } else {
    s = mkl_sparse_sp2m_64(op_a, descr_a, a.get(), op_b, descr_b, b.get(),
                           request, out);
  }
  throw_if_error(s, "mkl_sparse_sp2m");
}

template <typename J>
void set_mv_hint(const sparse_handle<J>& a, sparse_operation_t op,
                 matrix_descr descr, J expected_calls) {
  sparse_status_t s;
  if constexpr (!is_64<J>) {
    s = mkl_sparse_set_mv_hint(a.get(), op, descr, expected_calls);
  } else {
    s = mkl_sparse_set_mv_hint_64(a.get(), op, descr, expected_calls);
  }
  throw_if_error(s, "mkl_sparse_set_mv_hint");
}

template <typename J>
void set_mm_hint(const sparse_handle<J>& a, sparse_operation_t op,
                 matrix_descr descr, sparse_layout_t layout, J columns,
                 J expected_calls) {
  sparse_status_t s;
  if constexpr (!is_64<J>) {
    s = mkl_sparse_set_mm_hint(a.get(), op, descr, layout, columns,
                               expected_calls);
  } else {
    s = mkl_sparse_set_mm_hint_64(a.get(), op, descr, layout, columns,
                                  expected_calls);
  }
  throw_if_error(s, "mkl_sparse_set_mm_hint");
}

template <typename J>
void set_sv_hint(const sparse_handle<J>& a, sparse_operation_t op,
                 matrix_descr descr, J expected_calls) {
  sparse_status_t s;
  if constexpr (!is_64<J>) {
    s = mkl_sparse_set_sv_hint(a.get(), op, descr, expected_calls);
  } else {
    s = mkl_sparse_set_sv_hint_64(a.get(), op, descr, expected_calls);
  }
  throw_if_error(s, "mkl_sparse_set_sv_hint");
}

// mkl_sparse_order: sorts the column indices (and values) within each row;
// for a handle over user arrays, in those arrays.
template <typename J>
void order(const sparse_handle<J>& a) {
  sparse_status_t s;
  if constexpr (!is_64<J>) {
    s = mkl_sparse_order(a.get());
  } else {
    s = mkl_sparse_order_64(a.get());
  }
  throw_if_error(s, "mkl_sparse_order");
}

template <typename J>
void optimize(const sparse_handle<J>& a) {
  sparse_status_t s;
  if constexpr (!is_64<J>) {
    s = mkl_sparse_optimize(a.get());
  } else {
    s = mkl_sparse_optimize_64(a.get());
  }
  throw_if_error(s, "mkl_sparse_optimize");
}

} // namespace __mkl_iespblas

} // namespace spblas
