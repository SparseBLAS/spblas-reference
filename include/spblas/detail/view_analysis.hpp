#pragma once

#include <algorithm>
#include <cstddef>
#include <type_traits>
#include <utility>

#include <spblas/backend/concepts.hpp>
#include <spblas/backend/cpos.hpp>
#include <spblas/concepts.hpp>
#include <spblas/views/diagonal_view.hpp>
#include <spblas/views/inspectors.hpp>
#include <spblas/views/transposed_view.hpp>
#include <spblas/views/triangle_view.hpp>
#include <spblas/views/view.hpp>

//
// View composition analysis
//
//   __detail::analyze_view(M) reduces any stack of view::transpose,
//   view::conjugate, view::triangle and view::diagonal over a base matrix to
//   one canonical form,
//
//     M = op(region(base))
//
//   base      the innermost matrix that is not one of these views: a CSR/CSC
//             wrapper, a matrix_opt (whose optimization data stays
//             reachable), or another matrix; held by value.
//   region    full, lower, upper or diagonal, in base's own coordinates,
//             with `diag` saying how its diagonal is read (explicit, or
//             implied unit / zero).
//   op        transpose and / or conjugate (they commute), applied after
//             the region; known at compile time.
//
//   This is the form vendor libraries take (oneMKL: operation + matrix_descr
//   fill mode / diag, which refer to the handle, i.e. to base), and the
//   reference kernels rebuild the matrix from it (operand(),
//   represented_region()). Every spelling of the same matrix gives the same
//   analysis, whatever order the views are applied in and whether they are
//   named or temporaries:
//
//     view::transpose(view::triangle(A, lower))
//       == view::triangle(view::transpose(A), upper)   op = T, region lower
//     view::triangle(view::triangle(A, lower), upper)  region diagonal
//     view::conjugate(view::conjugate(A))              op = none
//
//   (for A a matrix_opt; view::transpose of a CSR/CSC wrapper reinterprets
//   it in the other format, so there base is that wrapper and op has no
//   transpose -- the represented matrix and represented_region() are the
//   same either way.)
//
//   Rules, applied from the innermost view outwards:
//     transpose(X)       toggles op's transpose
//     conjugate(X)       toggles op's conjugate (complex values only;
//                        view::conjugate of a real matrix is the matrix)
//     triangle(X, u, d)  region &= u, flipped to lower <-> upper when op
//                        transposes (the u triangle of op(Y) is op of the
//                        other triangle of Y)
//     diagonal(X, d)     region &= diagonal
//     with lower & upper = diagonal, and `d` replacing the diagonal handling
//     unless it is explicit_diagonal (an implied diagonal stays implied).
//
//   The views themselves stay as they are (nested, no base() for triangle /
//   diagonal); operations that support structure opt in by analyzing them.
//

namespace spblas {

namespace __detail {

enum class view_region { full, lower, upper, diagonal };

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
constexpr spblas::view::diag combine_diag(spblas::view::diag inner,
                                          spblas::view::diag outer) noexcept {
  return outer == spblas::view::diag::explicit_diagonal ? inner : outer;
}

template <typename Base, bool Transpose, bool Conjugate, bool Structured>
struct view_analysis {
  using base_type = Base;
  static constexpr bool transpose = Transpose;
  static constexpr bool conjugate = Conjugate;
  // Whether the stack contains a triangle or diagonal view.
  static constexpr bool structured = Structured;

  Base base;
  view_region region = view_region::full;
  spblas::view::diag diag = spblas::view::diag::explicit_diagonal;

  // The region of the represented matrix, op(region(base)).
  constexpr view_region represented_region() const noexcept {
    return transpose ? transposed_region(region) : region;
  }

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
};

template <matrix M>
auto analyze_view(M&& m) {
  using U = std::remove_cvref_t<M>;
  if constexpr (is_transposed_view_v<U>) {
    auto in = analyze_view(m.matrix());
    using In = decltype(in);
    return view_analysis<typename In::base_type, !In::transpose, In::conjugate,
                         In::structured>{in.base, in.region, in.diag};
  } else if constexpr (is_conjugated_view_v<U>) {
    auto in = analyze_view(m.base());
    using In = decltype(in);
    return view_analysis<typename In::base_type, In::transpose, !In::conjugate,
                         In::structured>{in.base, in.region, in.diag};
  } else if constexpr (is_triangle_view_v<U>) {
    auto in = analyze_view(m.matrix());
    using In = decltype(in);
    const view_region triangle = m.uplo() == spblas::view::uplo::lower
                                     ? view_region::lower
                                     : view_region::upper;
    const view_region in_base =
        In::transpose ? transposed_region(triangle) : triangle;
    return view_analysis<typename In::base_type, In::transpose, In::conjugate,
                         true>{in.base, intersect_regions(in.region, in_base),
                               combine_diag(in.diag, m.diag())};
  } else if constexpr (is_diagonal_view_v<U>) {
    auto in = analyze_view(m.matrix());
    using In = decltype(in);
    return view_analysis<typename In::base_type, In::transpose, In::conjugate,
                         true>{
        in.base, intersect_regions(in.region, view_region::diagonal),
        combine_diag(in.diag, m.diag())};
  } else {
    return view_analysis<U, false, false, false>{m};
  }
}

template <typename M>
using view_analysis_t = decltype(analyze_view(std::declval<M>()));

// Whether stored entry (i, j) of base is an entry of the view: inside
// `region`, and for a diagonal entry, read from storage (an implied diagonal
// replaces the stored one).
template <typename Index>
constexpr bool in_region(view_region region, spblas::view::diag diag, Index i,
                         Index j) noexcept {
  if (i == j) {
    return region == view_region::full ||
           diag == spblas::view::diag::explicit_diagonal;
  }
  switch (region) {
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

template <typename V>
tensor_offset_t<V> structured_view_size(const V& v) {
  auto a = analyze_view(v);
  using B = typename decltype(a)::base_type;
  tensor_offset_t<V> count = 0;
  if constexpr (__backend::row_iterable<B>) {
    for (auto&& [i, row] : __backend::rows(a.base)) {
      for (auto&& [j, _] : row) {
        count += in_region(a.region, a.diag, i, j) ? 1 : 0;
      }
    }
  } else if constexpr (__backend::column_iterable<B>) {
    for (auto&& [j, column] : __backend::columns(a.base)) {
      for (auto&& [i, _] : column) {
        count += in_region(a.region, a.diag, i, j) ? 1 : 0;
      }
    }
  } else {
    static_assert(__backend::row_iterable<B> || __backend::column_iterable<B>,
                  "the size of a triangle / diagonal view needs a row- or "
                  "column-iterable matrix");
  }
  // An implied unit diagonal is part of the represented matrix, so it
  // counts, one entry per diagonal position (structured regions always
  // contain the diagonal); an implied zero diagonal adds nothing.
  if (a.diag == spblas::view::diag::implicit_unit) {
    const auto shape = __backend::shape(a.base);
    count += tensor_offset_t<V>(std::min<std::size_t>(shape[0], shape[1]));
  }
  return count;
}

// A stack of views containing a triangle or diagonal view.
template <typename M>
concept structured_view =
    requires { typename view_analysis_t<M>; } && view_analysis_t<M>::structured;

} // namespace __detail

} // namespace spblas
