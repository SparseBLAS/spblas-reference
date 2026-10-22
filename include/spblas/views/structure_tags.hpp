#pragma once

#include <concepts>
#include <type_traits>
#include <utility>

// Compile-time arguments of the structure views (view::triangle,
// view::diagonal, view::symmetric, view::hermitian):
//
//   view::uplo::lower, view::uplo::upper
//   view::diag::explicit_diagonal, view::diag::implicit_unit,
//   view::diag::implicit_zero
//
// Each value is an object of its own type, so the matrix a view stack
// represents is known at compile time (see detail/view_analysis.hpp), and
// unsupported cases are compile errors. The SparseBLAS paper lists these
// values (with `explicit` for explicit_diagonal, a C++ keyword) but does not
// say whether they are enumerators or tags.
//
// To choose a triangle at run time, view::with_uplo(is_lower, f) calls
// f(view::uplo::lower) or f(view::uplo::upper).

namespace spblas {

namespace view {

namespace uplo {

struct lower_t {};
struct upper_t {};

inline constexpr lower_t lower{};
inline constexpr upper_t upper{};

} // namespace uplo

namespace diag {

// The diagonal is read from storage.
struct explicit_diagonal_t {};
// The diagonal is implied to be 1; stored diagonal entries are ignored.
struct implicit_unit_t {};
// The diagonal is implied to be 0; stored diagonal entries are ignored.
struct implicit_zero_t {};

inline constexpr explicit_diagonal_t explicit_diagonal{};
inline constexpr implicit_unit_t implicit_unit{};
inline constexpr implicit_zero_t implicit_zero{};

} // namespace diag

template <typename T>
concept uplo_tag = std::same_as<std::remove_cvref_t<T>, uplo::lower_t> ||
                   std::same_as<std::remove_cvref_t<T>, uplo::upper_t>;

template <typename T>
concept diag_tag =
    std::same_as<std::remove_cvref_t<T>, diag::explicit_diagonal_t> ||
    std::same_as<std::remove_cvref_t<T>, diag::implicit_unit_t> ||
    std::same_as<std::remove_cvref_t<T>, diag::implicit_zero_t>;

// Calls f(uplo::lower) if `is_lower`, else f(uplo::upper); both calls must
// return the same type.
template <typename F>
decltype(auto) with_uplo(bool is_lower, F&& f) {
  if (is_lower) {
    return std::forward<F>(f)(uplo::lower);
  }
  return std::forward<F>(f)(uplo::upper);
}

} // namespace view

namespace __detail {

// Regions of a matrix, in its own coordinates.
enum class view_region { full, lower, upper, diagonal };

// How a structure view reads the diagonal.
enum class diag_kind { explicit_diagonal, implicit_unit, implicit_zero };

template <spblas::view::uplo_tag U>
inline constexpr view_region region_of_v =
    std::same_as<std::remove_cvref_t<U>, spblas::view::uplo::lower_t>
        ? view_region::lower
        : view_region::upper;

template <spblas::view::diag_tag D>
inline constexpr diag_kind diag_kind_v =
    std::same_as<std::remove_cvref_t<D>,
                 spblas::view::diag::explicit_diagonal_t>
        ? diag_kind::explicit_diagonal
    : std::same_as<std::remove_cvref_t<D>, spblas::view::diag::implicit_unit_t>
        ? diag_kind::implicit_unit
        : diag_kind::implicit_zero;

// Tag types for the values above, to rebuild views from an analysis.
template <view_region R>
  requires(R == view_region::lower || R == view_region::upper)
using uplo_tag_t =
    std::conditional_t<R == view_region::lower, spblas::view::uplo::lower_t,
                       spblas::view::uplo::upper_t>;

template <diag_kind D>
using diag_tag_t =
    std::conditional_t<D == diag_kind::explicit_diagonal,
                       spblas::view::diag::explicit_diagonal_t,
                       std::conditional_t<D == diag_kind::implicit_unit,
                                          spblas::view::diag::implicit_unit_t,
                                          spblas::view::diag::implicit_zero_t>>;

} // namespace __detail

} // namespace spblas
