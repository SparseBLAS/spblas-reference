#pragma once

#include <spblas/detail/mdspan.hpp>
#include <spblas/views/conjugated_view.hpp>
#include <spblas/views/csc_view.hpp>
#include <spblas/views/csr_view.hpp>
#include <spblas/views/matrix_opt.hpp>
#include <spblas/views/scaled_view.hpp>

namespace spblas {

namespace __detail {

// Inspector for csr_wrapper

template <typename T>
struct is_instantiation_of_csr_wrapper {
  static constexpr bool value = false;
};

template <typename T, std::integral I, std::integral O>
struct is_instantiation_of_csr_wrapper<csr_wrapper<T, I, O>> {
  static constexpr bool value = true;
};

template <typename T>
static constexpr bool is_csr_wrapper_v =
    is_instantiation_of_csr_wrapper<std::remove_cvref_t<T>>::value;

// Deprecated name, kept until every backend has migrated.
template <typename T>
static constexpr bool is_csr_view_v = is_csr_wrapper_v<T>;

// Inspector for csc_wrapper

template <typename T>
struct is_instantiation_of_csc_wrapper {
  static constexpr bool value = false;
};

template <typename T, std::integral I, std::integral O>
struct is_instantiation_of_csc_wrapper<csc_wrapper<T, I, O>> {
  static constexpr bool value = true;
};

template <typename T>
static constexpr bool is_csc_wrapper_v =
    is_instantiation_of_csc_wrapper<std::remove_cvref_t<T>>::value;

// Deprecated name, kept until every backend has migrated.
template <typename T>
static constexpr bool is_csc_view_v = is_csc_wrapper_v<T>;

// Inspector for mdspan

template <typename T>
struct is_matrix_instantiation_of_mdspan {
  static constexpr bool value = false;
};

template <typename T, typename Extents, typename LayoutPolicy,
          typename AccessorPolicy>
  requires(Extents::rank() == 2)
struct is_matrix_instantiation_of_mdspan<
    __mdspan::mdspan<T, Extents, LayoutPolicy, AccessorPolicy>> {
  static constexpr bool value = true;
};

template <typename T>
static constexpr bool is_matrix_mdspan_v =
    is_matrix_instantiation_of_mdspan<std::remove_cvref_t<T>>::value;

template <typename T>
struct is_instantiation_of_scaled_view {
  static constexpr bool value = false;
};

template <typename S, typename T>
struct is_instantiation_of_scaled_view<scaled_view<S, T>> {
  static constexpr bool value = true;
};

template <typename T>
static constexpr bool is_scaled_view_v =
    is_instantiation_of_scaled_view<std::remove_cvref_t<T>>::value;

template <typename T>
static constexpr bool is_scaled_view_matrix_v =
    is_scaled_view_v<T> && matrix<decltype(std::declval<T>().base())>;

template <typename T>
struct is_instantiation_of_conjugated_view {
  static constexpr bool value = false;
};

template <typename T>
struct is_instantiation_of_conjugated_view<conjugated_view<T>> {
  static constexpr bool value = true;
};

template <typename T>
static constexpr bool is_conjugated_view_v =
    is_instantiation_of_conjugated_view<std::remove_cvref_t<T>>::value;

template <typename T>
static constexpr bool is_conjugated_view_matrix_v =
    is_conjugated_view_v<T> && matrix<decltype(std::declval<T>().base())>;

template <typename T>
struct is_instantiation_of_matrix_opt {
  static constexpr bool value = false;
};

template <typename T>
struct is_instantiation_of_matrix_opt<matrix_opt<T>> {
  static constexpr bool value = true;
};

template <typename T>
static constexpr bool is_matrix_opt_v =
    is_instantiation_of_matrix_opt<std::remove_cvref_t<T>>::value;

} // namespace __detail

} // namespace spblas
