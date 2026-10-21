#pragma once

#include <type_traits>

#include <spblas/backend/concepts.hpp>
#include <spblas/backend/cpos.hpp>
#include <spblas/concepts.hpp>
#include <spblas/detail/index.hpp>
#include <spblas/views/view_base.hpp>

namespace spblas {

// The transpose of a matrix that cannot simply be reinterpreted in another
// format: a matrix_opt (whose optimization data must stay attached), a
// conjugated view, or a dense matrix. view::transpose of a CSR/CSC wrapper
// returns the other format instead.
//
// Rows of the view are the columns of `matrix()` and vice versa, and lookups
// swap their indices, so reference kernels see A^T through the CPOs. Vendors
// can instead use `matrix()` with a transpose flag.
//
// Like triangle_view this has no `base()`: it changes which matrix is
// represented, so it must not be unwrapped transparently by
// `get_ultimate_base()` / `has_csr_base`. Operations that support it work
// through the CPOs or match `is_transposed_view_v`; others fail to compile.
template <matrix M>
class transposed_view : public view_base {
public:
  using scalar_type = tensor_scalar_t<M>;
  using scalar_reference = tensor_scalar_reference_t<M>;
  using index_type = tensor_index_t<M>;
  using offset_type = tensor_offset_t<M>;

  explicit transposed_view(M matrix) : matrix_(matrix) {}

  M matrix() const {
    return matrix_;
  }

  auto shape() const noexcept {
    auto s = __backend::shape(matrix_);
    return index<index_type>(s[1], s[0]);
  }

  auto size() const noexcept {
    return __backend::size(matrix_);
  }

private:
  friend auto tag_invoke(__backend::size_fn_, const transposed_view& t) {
    return t.size();
  }

  friend auto tag_invoke(__backend::shape_fn_, const transposed_view& t) {
    return t.shape();
  }

  friend auto tag_invoke(__backend::rows_fn_, const transposed_view& t)
    requires(__backend::column_iterable<M>)
  {
    return __backend::columns(t.matrix_);
  }

  friend auto tag_invoke(__backend::columns_fn_, const transposed_view& t)
    requires(__backend::row_iterable<M>)
  {
    return __backend::rows(t.matrix_);
  }

  friend auto tag_invoke(__backend::lookup_row_fn_, const transposed_view& t,
                         index_type row_index)
    requires(__backend::column_lookupable<M>)
  {
    return __backend::lookup_column(t.matrix_, row_index);
  }

  friend auto tag_invoke(__backend::lookup_column_fn_, const transposed_view& t,
                         index_type column_index)
    requires(__backend::row_lookupable<M>)
  {
    return __backend::lookup_row(t.matrix_, column_index);
  }

  friend decltype(auto) tag_invoke(__backend::lookup_fn_,
                                   const transposed_view& t, index_type i,
                                   index_type j)
    requires(__backend::lookupable<M>)
  {
    return __backend::lookup(t.matrix_, j, i);
  }

  M matrix_;
};

namespace __detail {

template <typename T>
struct is_instantiation_of_transposed_view {
  static constexpr bool value = false;
};

template <typename M>
struct is_instantiation_of_transposed_view<transposed_view<M>> {
  static constexpr bool value = true;
};

template <typename T>
static constexpr bool is_transposed_view_v =
    is_instantiation_of_transposed_view<std::remove_cvref_t<T>>::value;

} // namespace __detail

} // namespace spblas
