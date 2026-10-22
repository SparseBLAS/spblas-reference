#pragma once

#include <algorithm>
#include <complex>
#include <cstddef>
#include <type_traits>
#include <utility>

#include <spblas/backend/concepts.hpp>
#include <spblas/backend/cpos.hpp>
#include <spblas/concepts.hpp>
#include <spblas/detail/type_traits.hpp>
#include <spblas/views/inspectors.hpp>
#include <spblas/views/structured_matrix_view.hpp>
#include <spblas/views/transposed_view.hpp>
#include <spblas/views/view.hpp>

//
// View composition analysis
//
//   __detail::analyze_view(M) reduces any stack of view::transpose,
//   view::conjugate, view::triangle, view::diagonal, view::symmetric and
//   view::hermitian over a base matrix to one canonical form, known entirely
//   at compile time (view::uplo / view::diag are tags):
//
//     M = op(structure(region(base)))
//
//   base       the innermost matrix that is not one of these views: a CSR/CSC
//              wrapper, a matrix_opt (whose optimization data stays
//              reachable), or another matrix; held by value.
//   region     full, lower, upper or diagonal, in base's own coordinates,
//              with `diag` saying how its diagonal is read (explicit, or
//              implied unit / zero).
//   structure  general, symmetric (T + S^T) or hermitian (T + S^H, real part
//              of the diagonal), completing a lower / upper region.
//   op         transpose and / or conjugate (they commute), applied last.
//              Symmetric structures absorb a transpose; for hermitian ones a
//              transpose is a conjugate. A diagonal region drops transpose.
//
//   This is the form vendor libraries take (oneMKL: operation + matrix_descr
//   type / fill mode / diag, which refer to the handle, i.e. to base), and
//   the reference kernels read the represented matrix from it
//   (for_each_represented_entry, operand()). Every spelling of the same
//   matrix gives the same analysis, in any order, for named views as for
//   temporaries:
//
//     view::transpose(view::triangle(A, lower))
//       == view::triangle(view::transpose(A), upper)    op = T, region lower
//     view::triangle(view::triangle(A, lower), upper)   region diagonal
//     view::triangle(view::symmetric(A, lower), upper)  op = T, region lower
//     view::symmetric(view::transpose(A), lower)        symmetric, upper
//
//   (for A a matrix_opt; view::transpose of a CSR/CSC wrapper reinterprets it
//   in the other format, so there base is that wrapper.)
//
//   Rules, applied from the innermost view outwards (u' = u flipped to the
//   other triangle when op transposes, u as is otherwise):
//     transpose(X)       toggles op's transpose (see `structure` above)
//     conjugate(X)       toggles op's conjugate (complex values only;
//                        view::conjugate of a real matrix is the matrix)
//     triangle(X, u, d)  general X: region &= u'.
//                        symmetric / hermitian X with region r: the u
//                        triangle is r's triangle (u == r) or its transpose /
//                        conjugate transpose (u != r); structure general.
//     diagonal(X, d)     region &= diagonal, structure general
//     symmetric(X, u, d) general X: region &= u', structure symmetric
//     hermitian(X, u, d) likewise, hermitian (symmetric for real values)
//     with lower & upper = diagonal, and `d` replacing the diagonal handling
//     unless it is explicit_diagonal (an implied diagonal stays implied).
//
//   Not representable (rejected: no analysis, so no operation accepts it):
//     - the explicit diagonal of a hermitian structure read by a triangle /
//       diagonal view, or a hermitian diagonal region (they would need the
//       real part of the diagonal outside a hermitian structure);
//     - mixing symmetric and hermitian structures.
//
//   The views themselves stay as they are (nested, no base() for structure
//   views); operations that support structure opt in by analyzing them.
//

namespace spblas {

namespace __detail {

enum class view_structure { general, symmetric, hermitian };

// The canonical form of a view stack (a structural type, usable as a
// template argument).
struct view_desc {
  bool transpose = false;
  bool conjugate = false;
  view_region region = view_region::full;
  diag_kind diag = diag_kind::explicit_diagonal;
  view_structure structure = view_structure::general;
  // Whether the stack contains a structure view.
  bool structured = false;
  // Whether the stack is representable (see above).
  bool valid = true;

  friend constexpr bool operator==(const view_desc&,
                                   const view_desc&) = default;
};

constexpr view_region transposed_region(view_region r) noexcept {
  switch (r) {
  case view_region::lower:
    return view_region::upper;
  case view_region::upper:
    return view_region::lower;
  default:
    return r;
  }
}

constexpr view_region intersect_regions(view_region a, view_region b) noexcept {
  if (a == view_region::full || a == b) {
    return b;
  }
  if (b == view_region::full) {
    return a;
  }
  // lower & upper, or either with diagonal.
  return view_region::diagonal;
}

// An outer view's diagonal handling replaces the inner one's unless it reads
// the (possibly implied) diagonal as it is.
constexpr diag_kind combine_diag(diag_kind inner, diag_kind outer) noexcept {
  return outer == diag_kind::explicit_diagonal ? inner : outer;
}

constexpr view_desc invalid_desc() noexcept {
  view_desc d;
  d.valid = false;
  return d;
}

// A diagonal region is its own transpose, and symmetric / hermitian
// completions of it are the diagonal itself.
constexpr view_desc normalized(view_desc d) noexcept {
  if (d.region == view_region::diagonal) {
    d.transpose = false;
    d.structure = view_structure::general;
  }
  return d;
}

constexpr view_desc apply_transpose(view_desc d) noexcept {
  switch (d.structure) {
  case view_structure::general:
    d.transpose = !d.transpose;
    break;
  case view_structure::symmetric: // S^T = S
    break;
  case view_structure::hermitian: // H^T = conj(H)
    d.conjugate = !d.conjugate;
    break;
  }
  return normalized(d);
}

constexpr view_desc apply_conjugate(view_desc d) noexcept {
  d.conjugate = !d.conjugate;
  return d;
}

constexpr view_desc apply_triangle(view_desc d, view_region u,
                                   diag_kind diag) noexcept {
  if (!d.valid) {
    return d;
  }
  d.structured = true;
  if (d.structure == view_structure::general) {
    d.region =
        intersect_regions(d.region, d.transpose ? transposed_region(u) : u);
  } else {
    // The u triangle of op(S(r)), S symmetric or hermitian, r the stored
    // triangle: r's triangle itself, or its transpose (S^H's: conjugate
    // transpose) when u is the other triangle. A hermitian diagonal would
    // need its real part, so only an implied diagonal is representable.
    if (d.structure == view_structure::hermitian &&
        diag == diag_kind::explicit_diagonal) {
      return invalid_desc();
    }
    if (u != d.region) {
      d.transpose = true;
      if (d.structure == view_structure::hermitian) {
        d.conjugate = !d.conjugate;
      }
    }
    d.structure = view_structure::general;
  }
  d.diag = combine_diag(d.diag, diag);
  return normalized(d);
}

constexpr view_desc apply_diagonal(view_desc d, diag_kind diag) noexcept {
  if (!d.valid) {
    return d;
  }
  if (d.structure == view_structure::hermitian &&
      diag == diag_kind::explicit_diagonal) {
    return invalid_desc();
  }
  d.structured = true;
  d.region = intersect_regions(d.region, view_region::diagonal);
  d.diag = combine_diag(d.diag, diag);
  return normalized(d);
}

constexpr view_desc apply_symmetric(view_desc d, view_region u, diag_kind diag,
                                    bool hermitian) noexcept {
  if (!d.valid) {
    return d;
  }
  d.structured = true;
  const view_structure s =
      hermitian ? view_structure::hermitian : view_structure::symmetric;
  if (d.structure != view_structure::general) {
    // S(S(M, r), u) == S(M, r); mixing symmetric and hermitian is not
    // representable.
    if (d.structure != s) {
      return invalid_desc();
    }
    d.diag = combine_diag(d.diag, diag);
    return d;
  }
  d.region =
      intersect_regions(d.region, d.transpose ? transposed_region(u) : u);
  d.diag = combine_diag(d.diag, diag);
  if (d.region == view_region::diagonal) {
    // The completion of a diagonal is the diagonal; a hermitian one would
    // need its real part.
    if (hermitian && d.diag == diag_kind::explicit_diagonal) {
      return invalid_desc();
    }
    return normalized(d);
  }
  d.structure = s;
  if (d.transpose) {
    // S(Y^T) == S(Y) on the flipped triangle; H(Y^T) == conj(H(Y)).
    d.transpose = false;
    if (hermitian) {
      d.conjugate = !d.conjugate;
    }
  }
  return d;
}

// A conjugated matrix (not vector) view. (Nested so that matrix<M> is only
// checked for conjugated views: checking it for an unsupported structure
// view would instantiate its size().)
template <typename M>
constexpr bool is_conjugated_matrix_view_() {
  if constexpr (is_conjugated_view_v<M>) {
    return matrix<M>;
  } else {
    return false;
  }
}

// The innermost matrix of a view stack.
template <typename M>
struct view_base_type {
  using type = std::remove_cvref_t<M>;
};

template <typename M>
using view_base_t = typename view_base_type<std::remove_cvref_t<M>>::type;

template <typename M>
  requires(is_transposed_view_v<M> || is_structure_view_v<M>)
struct view_base_type<M> {
  using type = view_base_t<decltype(std::declval<const M&>().matrix())>;
};

template <typename M>
  requires(is_conjugated_matrix_view_<M>())
struct view_base_type<M> {
  using type = view_base_t<decltype(std::declval<const M&>().base())>;
};

template <typename M>
view_base_t<M> view_base(const M& m) {
  using U = std::remove_cvref_t<M>;
  if constexpr (is_transposed_view_v<U> || is_structure_view_v<U>) {
    return view_base(m.matrix());
  } else if constexpr (is_conjugated_matrix_view_<U>()) {
    return view_base(m.base());
  } else {
    return m;
  }
}

template <typename M>
constexpr view_desc describe_view() {
  using U = std::remove_cvref_t<M>;
  constexpr bool complex = is_std_complex_v<tensor_scalar_t<view_base_t<U>>>;
  if constexpr (is_transposed_view_v<U>) {
    return apply_transpose(
        describe_view<decltype(std::declval<const U&>().matrix())>());
  } else if constexpr (is_conjugated_matrix_view_<U>()) {
    return apply_conjugate(
        describe_view<decltype(std::declval<const U&>().base())>());
  } else if constexpr (is_structure_view_v<U>) {
    constexpr view_desc in =
        describe_view<decltype(std::declval<const U&>().matrix())>();
    constexpr diag_kind diag = diag_kind_v<typename U::diag_type>;
    if constexpr (U::kind == structure_kind::diagonal) {
      return apply_diagonal(in, diag);
    } else {
      constexpr view_region u = region_of_v<typename U::uplo_type>;
      if constexpr (U::kind == structure_kind::triangle) {
        return apply_triangle(in, u, diag);
      } else {
        return apply_symmetric(in, u, diag,
                               U::kind == structure_kind::hermitian && complex);
      }
    }
  } else {
    return view_desc{};
  }
}

template <typename Base, view_desc Desc>
struct view_analysis {
  using base_type = Base;
  static constexpr view_desc desc = Desc;
  static constexpr bool transpose = Desc.transpose;
  static constexpr bool conjugate = Desc.conjugate;
  static constexpr bool structured = Desc.structured;
  static constexpr view_region region = Desc.region;
  static constexpr diag_kind diag = Desc.diag;
  static constexpr view_structure structure = Desc.structure;

  // The region of the represented matrix, op(region(base)) (for general
  // structures).
  static constexpr view_region represented_region =
      transpose ? transposed_region(region) : region;

  Base base;

  // op(base): base with the stack's transpose / conjugate but without its
  // structure, as views the reference kernels iterate through the CPOs.
  auto operand() const {
    auto conjugated = [&] {
      if constexpr (conjugate) {
        return spblas::view::conjugate(base);
      } else {
        return base;
      }
    }();
    if constexpr (transpose) {
      return spblas::view::transpose(conjugated);
    } else {
      return conjugated;
    }
  }

  // Shape of the represented matrix.
  auto shape() const {
    auto s = __backend::shape(base);
    if constexpr (transpose) {
      return decltype(s)(s[1], s[0]);
    } else {
      return s;
    }
  }
};

// A view stack the analysis can represent.
template <typename M>
concept analyzable_view = matrix<M> && describe_view<M>().valid;

template <analyzable_view M>
auto analyze_view(const M& m) {
  return view_analysis<view_base_t<M>, describe_view<M>()>{view_base(m)};
}

template <typename M>
using view_analysis_t = decltype(analyze_view(std::declval<M>()));

// A view stack containing a structure view.
template <typename M>
concept structured_view = analyzable_view<M> && describe_view<M>().structured;

// Whether stored entry (i, j) of base is read by the analysis: inside
// `region`, and for a diagonal entry, read from storage.
template <view_desc D, typename Index>
constexpr bool reads_entry(Index i, Index j) noexcept {
  if (i == j) {
    return D.region == view_region::full ||
           D.diag == diag_kind::explicit_diagonal;
  }
  switch (D.region) {
  case view_region::lower:
    return j < i;
  case view_region::upper:
    return j > i;
  case view_region::diagonal:
    return false;
  default:
    return true;
  }
}

// Calls f(row, column, value) for every entry of the matrix an analysis
// represents (mirrored entries of symmetric / hermitian structures and
// implied unit diagonal entries included, implied zeros not), in no
// particular order. One pass over base's stored entries.
template <typename Analysis, typename F>
void for_each_represented_entry(const Analysis& a, F&& f) {
  using B = typename Analysis::base_type;
  using T = tensor_scalar_t<B>;
  constexpr view_desc D = Analysis::desc;
  constexpr bool complex = is_std_complex_v<T>;

  auto visit = [&](auto i, auto j, T v) {
    if (!reads_entry<D>(i, j)) {
      return;
    }
    if constexpr (D.conjugate && complex) {
      v = std::conj(v);
    }
    if constexpr (D.structure == view_structure::general) {
      if constexpr (D.transpose) {
        f(j, i, v);
      } else {
        f(i, j, v);
      }
    } else if constexpr (D.structure == view_structure::symmetric) {
      f(i, j, v);
      if (i != j) {
        f(j, i, v);
      }
    } else {
      if (i == j) {
        if constexpr (complex) {
          f(i, j, T(v.real()));
        } else {
          f(i, j, v);
        }
      } else {
        f(i, j, v);
        if constexpr (complex) {
          f(j, i, std::conj(v));
        } else {
          f(j, i, v);
        }
      }
    }
  };

  if constexpr (__backend::row_iterable<B>) {
    for (auto&& [i, row] : __backend::rows(a.base)) {
      for (auto&& [j, v] : row) {
        visit(i, j, v);
      }
    }
  } else if constexpr (__backend::column_iterable<B>) {
    for (auto&& [j, column] : __backend::columns(a.base)) {
      for (auto&& [i, v] : column) {
        visit(i, j, v);
      }
    }
  } else {
    static_assert(__backend::row_iterable<B> || __backend::column_iterable<B>,
                  "a structure view needs a row- or column-iterable matrix");
  }

  if constexpr (D.diag == diag_kind::implicit_unit) {
    const auto s = __backend::shape(a.base);
    using I = tensor_index_t<B>;
    const I n = I(std::min<std::size_t>(s[0], s[1]));
    for (I k = 0; k < n; k++) {
      f(k, k, T(1));
    }
  }
}

template <typename V>
tensor_offset_t<V> structured_view_size(const V& v) {
  static_assert(analyzable_view<V>,
                "this composition of structure views is not supported (see "
                "detail/view_analysis.hpp)");
  tensor_offset_t<V> count = 0;
  for_each_represented_entry(analyze_view(v),
                             [&](auto, auto, auto&&) { count++; });
  return count;
}

} // namespace __detail

} // namespace spblas
