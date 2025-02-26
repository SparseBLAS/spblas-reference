#pragma once

#include <spblas/backend/concepts.hpp>
#include <spblas/backend/cpos.hpp>
#include <spblas/concepts.hpp>

namespace spblas {

template <matrix M>
  requires(view<M>)
class matrix_opt : public view_base {
public:
  using scalar_type = tensor_scalar_t<M>;
  using scalar_reference = tensor_scalar_reference_t<M>;
  using index_type = tensor_index_t<M>;
  using offset_type = tensor_offset_t<M>;

  matrix_opt(M matrix) : matrix_(matrix) {}

  auto shape() const noexcept {
    return __backend::shape(base());
  }

  index_type size() const noexcept {
    return __backend::size(base());
  }

  auto base() {
    return matrix_;
  }

  auto base() const {
    return matrix_;
  }

private:
  friend auto tag_invoke(__backend::size_fn_, matrix_opt matrix) {
    return matrix.size();
  }

  friend auto tag_invoke(__backend::shape_fn_, matrix_opt matrix) {
    return matrix.shape();
  }

  friend scalar_reference tag_invoke(__backend::lookup_fn_, matrix_opt matrix,
                                     index_type i, index_type j)
    requires(__backend::lookupable<M>)
  {
    return __backend::lookup(matrix.base(), i, j);
  }

  friend auto tag_invoke(__backend::rows_fn_, matrix_opt matrix)
    requires(__backend::row_iterable<M>)
  {
    return __backend::rows(matrix.base());
  }

  friend auto tag_invoke(__backend::lookup_row_fn_, matrix_opt matrix,
                         index_type row_index)
    requires(__backend::row_lookupable<M>)
  {
    return __backend::lookup_row(matrix.base(), row_index);
  }

public:
  M matrix_;
};

namespace __detail {

template <typename T>
struct is_instantiation_of_matrix_opt_view {
  static constexpr bool value = false;
};

template <typename T>
struct is_instantiation_of_matrix_opt_view<matrix_opt<T>> {
  static constexpr bool value = true;
};

template <typename T>
static constexpr bool is_matrix_opt_view_v =
    is_instantiation_of_matrix_opt_view<std::remove_cvref_t<T>>::value;

} // namespace __detail

} // namespace spblas
