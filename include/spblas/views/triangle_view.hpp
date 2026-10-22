#pragma once

#include <type_traits>

#include <spblas/backend/concepts.hpp>
#include <spblas/backend/cpos.hpp>
#include <spblas/concepts.hpp>
#include <spblas/detail/cached_size.hpp>
#include <spblas/views/view_base.hpp>

namespace spblas {

namespace __detail {

// Number of stored entries inside a triangle / diagonal view, whatever views
// it is composed of; defined in detail/view_analysis.hpp.
template <typename V>
tensor_offset_t<V> structured_view_size(const V& v);

} // namespace __detail

namespace view {

enum class uplo { lower, upper };

// implicit_zero is part of the API, but no backend supports it yet; backends
// throw when it is used.
enum class diag { explicit_diagonal, implicit_unit, implicit_zero };

} // namespace view

// The `uplo` triangle of a matrix. The diagonal is read from storage
// (`explicit_diagonal`) or implied (`implicit_unit`, `implicit_zero`), in
// which case stored diagonal entries are ignored.
//
// Unlike scaled/conjugated views this has no `base()`: it changes which
// matrix is represented, so it must not be unwrapped transparently by
// `get_ultimate_base()` / `has_csr_base`. Operations that support it match
// `is_triangle_view_v` and use `matrix()`; all others fail to compile.
template <matrix M>
class triangle_view : public view_base {
public:
  using scalar_type = tensor_scalar_t<M>;
  using scalar_reference = tensor_scalar_reference_t<M>;
  using index_type = tensor_index_t<M>;
  using offset_type = tensor_offset_t<M>;

  triangle_view(M matrix, view::uplo uplo, view::diag diag)
      : matrix_(matrix), uplo_(uplo), diag_(diag) {}

  M matrix() const {
    return matrix_;
  }

  view::uplo uplo() const noexcept {
    return uplo_;
  }

  view::diag diag() const noexcept {
    return diag_;
  }

  auto shape() const noexcept {
    return __backend::shape(matrix_);
  }

  // Number of entries of the represented triangle: the stored entries of
  // `matrix()` inside it, with the diagonal from storage, or one entry per
  // diagonal position for implicit_unit, or none for implicit_zero. O(nnz) on
  // the first call, then cached.
  offset_type size() const {
    return size_.get([&] { return __detail::structured_view_size(*this); });
  }

private:
  friend auto tag_invoke(__backend::size_fn_, const triangle_view& t) {
    return t.size();
  }

  friend auto tag_invoke(__backend::shape_fn_, const triangle_view& t) {
    return t.shape();
  }

  M matrix_;
  view::uplo uplo_;
  view::diag diag_;
  __detail::cached_size<offset_type> size_;
};

namespace __detail {

template <typename T>
struct is_instantiation_of_triangle_view {
  static constexpr bool value = false;
};

template <typename M>
struct is_instantiation_of_triangle_view<triangle_view<M>> {
  static constexpr bool value = true;
};

template <typename T>
static constexpr bool is_triangle_view_v =
    is_instantiation_of_triangle_view<std::remove_cvref_t<T>>::value;

} // namespace __detail

} // namespace spblas
