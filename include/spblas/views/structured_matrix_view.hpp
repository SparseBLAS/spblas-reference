#pragma once

#include <type_traits>

#include <spblas/backend/concepts.hpp>
#include <spblas/backend/cpos.hpp>
#include <spblas/concepts.hpp>
#include <spblas/detail/cached_size.hpp>
#include <spblas/views/structure_tags.hpp>
#include <spblas/views/view_base.hpp>

namespace spblas {

namespace __detail {

// Number of entries of the matrix a structure view represents; defined in
// detail/view_analysis.hpp.
template <typename V>
tensor_offset_t<V> structured_view_size(const V& v);

} // namespace __detail

// The kinds of structure view.
enum class structure_kind { triangle, diagonal, symmetric, hermitian };

// A matrix read through a structure (SparseBLAS paper, views table), with
// the `Uplo` triangle and the diagonal handling `Diag` as types:
//
//   triangle_view<M, U, D>   the U triangle of M
//   diagonal_view<M, D>      the diagonal of M (Uplo is void)
//   symmetric_view<M, U, D>  the symmetric matrix T + S^T, T the U triangle
//                            of M and S its strict part
//   hermitian_view<M, U, D>  T + S^H, using only the real part of the
//                            stored diagonal (as LAPACK's ?hemv)
//
// The diagonal is read from storage (explicit_diagonal) or implied
// (implicit_unit, implicit_zero; stored diagonal entries are then ignored).
//
// These views have no `base()`: they change which matrix is represented, so
// they must not be unwrapped transparently by `get_ultimate_base()` /
// `has_csr_base`. Operations that support them analyze the view stack
// (__detail::analyze_view, detail/view_analysis.hpp); all others fail to
// compile.
template <structure_kind Kind, matrix M, typename Uplo, typename Diag>
  requires((Kind == structure_kind::diagonal ? std::is_void_v<Uplo>
                                             : view::uplo_tag<Uplo>) &&
           view::diag_tag<Diag>)
class structured_matrix_view : public view_base {
public:
  using scalar_type = tensor_scalar_t<M>;
  using scalar_reference = tensor_scalar_reference_t<M>;
  using index_type = tensor_index_t<M>;
  using offset_type = tensor_offset_t<M>;
  using matrix_type = M;
  using uplo_type = Uplo;
  using diag_type = Diag;

  static constexpr structure_kind kind = Kind;

  explicit structured_matrix_view(M matrix) : matrix_(matrix) {}

  M matrix() const {
    return matrix_;
  }

  auto shape() const noexcept {
    return __backend::shape(matrix_);
  }

  // Number of entries of the represented matrix: the stored entries of
  // matrix() it uses (strict-triangle entries twice for symmetric /
  // hermitian), with the diagonal from storage (explicit_diagonal), one
  // entry per diagonal position (implicit_unit), or none (implicit_zero).
  // O(nnz) on the first call, then cached.
  offset_type size() const {
    return size_.get([&] { return __detail::structured_view_size(*this); });
  }

private:
  // (Declared return type: concept checks must not instantiate size(), which
  // rejects unsupported compositions.)
  friend offset_type tag_invoke(__backend::size_fn_,
                                const structured_matrix_view& v) {
    return v.size();
  }

  friend auto tag_invoke(__backend::shape_fn_,
                         const structured_matrix_view& v) {
    return v.shape();
  }

  M matrix_;
  __detail::cached_size<offset_type> size_;
};

template <matrix M, view::uplo_tag U, view::diag_tag D>
using triangle_view = structured_matrix_view<structure_kind::triangle, M, U, D>;

template <matrix M, view::diag_tag D>
using diagonal_view =
    structured_matrix_view<structure_kind::diagonal, M, void, D>;

template <matrix M, view::uplo_tag U, view::diag_tag D>
using symmetric_view =
    structured_matrix_view<structure_kind::symmetric, M, U, D>;

template <matrix M, view::uplo_tag U, view::diag_tag D>
using hermitian_view =
    structured_matrix_view<structure_kind::hermitian, M, U, D>;

namespace __detail {

template <typename T>
struct structure_kind_of {
  static constexpr bool is_structure_view = false;
};

template <structure_kind Kind, typename M, typename U, typename D>
struct structure_kind_of<structured_matrix_view<Kind, M, U, D>> {
  static constexpr bool is_structure_view = true;
  static constexpr structure_kind kind = Kind;
};

template <typename T>
static constexpr bool is_structure_view_v =
    structure_kind_of<std::remove_cvref_t<T>>::is_structure_view;

template <typename T, structure_kind Kind>
constexpr bool is_structure_view_of_() {
  if constexpr (is_structure_view_v<T>) {
    return structure_kind_of<std::remove_cvref_t<T>>::kind == Kind;
  } else {
    return false;
  }
}

template <typename T>
static constexpr bool is_triangle_view_v =
    is_structure_view_of_<T, structure_kind::triangle>();

template <typename T>
static constexpr bool is_diagonal_view_v =
    is_structure_view_of_<T, structure_kind::diagonal>();

template <typename T>
static constexpr bool is_symmetric_view_v =
    is_structure_view_of_<T, structure_kind::symmetric>();

template <typename T>
static constexpr bool is_hermitian_view_v =
    is_structure_view_of_<T, structure_kind::hermitian>();

} // namespace __detail

} // namespace spblas
