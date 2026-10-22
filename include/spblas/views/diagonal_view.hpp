#pragma once

#include <type_traits>

#include <spblas/backend/concepts.hpp>
#include <spblas/backend/cpos.hpp>
#include <spblas/concepts.hpp>
#include <spblas/detail/cached_size.hpp>
#include <spblas/views/triangle_view.hpp>
#include <spblas/views/view_base.hpp>

namespace spblas {

// The diagonal of a matrix (not its strictly lower or upper triangle). The
// diagonal is read from storage (`explicit_diagonal`) or implied
// (`implicit_unit`: the identity, `implicit_zero`: the zero matrix).
//
// Like triangle_view this has no `base()`: it changes which matrix is
// represented, so it must not be unwrapped transparently. Operations that
// support it go through __detail::analyze_view (detail/view_analysis.hpp).
template <matrix M>
class diagonal_view : public view_base {
public:
  using scalar_type = tensor_scalar_t<M>;
  using scalar_reference = tensor_scalar_reference_t<M>;
  using index_type = tensor_index_t<M>;
  using offset_type = tensor_offset_t<M>;

  diagonal_view(M matrix, view::diag diag) : matrix_(matrix), diag_(diag) {}

  M matrix() const {
    return matrix_;
  }

  view::diag diag() const noexcept {
    return diag_;
  }

  auto shape() const noexcept {
    return __backend::shape(matrix_);
  }

  // Number of stored diagonal entries of `matrix()`; implied diagonal
  // entries are not counted. O(nnz) on the first call, then cached.
  offset_type size() const {
    return size_.get([&] { return __detail::structured_view_size(*this); });
  }

private:
  friend auto tag_invoke(__backend::size_fn_, const diagonal_view& d) {
    return d.size();
  }

  friend auto tag_invoke(__backend::shape_fn_, const diagonal_view& d) {
    return d.shape();
  }

  M matrix_;
  view::diag diag_;
  __detail::cached_size<offset_type> size_;
};

namespace __detail {

template <typename T>
struct is_instantiation_of_diagonal_view {
  static constexpr bool value = false;
};

template <typename M>
struct is_instantiation_of_diagonal_view<diagonal_view<M>> {
  static constexpr bool value = true;
};

template <typename T>
static constexpr bool is_diagonal_view_v =
    is_instantiation_of_diagonal_view<std::remove_cvref_t<T>>::value;

} // namespace __detail

} // namespace spblas
