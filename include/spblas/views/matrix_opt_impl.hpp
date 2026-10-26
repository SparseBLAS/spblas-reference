#pragma once

#include <memory>
#include <memory_resource>

#include <spblas/backend/concepts.hpp>
#include <spblas/backend/cpos.hpp>
#include <spblas/concepts.hpp>
#include <spblas/detail/allocator.hpp>
#include <spblas/detail/backend_state.hpp>
#include <spblas/views/properties.hpp>

namespace spblas {

// A matrix wrapper plus opaque, library-owned optimization data for it.
// Copies share the optimization data, which is released with the last copy.
// The data is allocated from the allocator given at construction (a
// spblas::allocator, or a std::pmr resource / polymorphic_allocator as its
// host role); the shared data itself comes from the host role.
//
// The matrix_opt keeps its own copy of the wrapper, shared by all copies of
// the matrix_opt (and the views over them), so properties set through any of
// them (e.g. `sort_matrix(state, A_opt)` setting `sorted`) are seen by all.
// The wrapper the matrix_opt was constructed from is not updated;
// `A.sync_properties(A_opt)` copies the properties back into it.
template <matrix M>
  requires(__detail::view<M> &&
           (__detail::is_csr_view_v<M> || __detail::is_csc_view_v<M>) )
class matrix_opt<M> : public view_base {
public:
  using scalar_type = tensor_scalar_t<M>;
  using scalar_reference = tensor_scalar_reference_t<M>;
  using index_type = tensor_index_t<M>;
  using offset_type = tensor_offset_t<M>;

  explicit matrix_opt(M matrix, spblas::allocator allocator = {})
      : allocator_(allocator) {
    auto data = std::allocate_shared<shared_data_>(
        std::pmr::polymorphic_allocator<shared_data_>(
            allocator_.host().resource()),
        matrix);
    matrix_ = &data->matrix;
    state_ = std::shared_ptr<__detail::backend_matrix_opt_state_t>(
        data, &data->state);
  }

  auto shape() const noexcept {
    return __backend::shape(base());
  }

  offset_type size() const noexcept {
    return __backend::size(base());
  }

  M base() const {
    return *matrix_;
  }

  property_list properties() const noexcept {
    return matrix_->properties();
  }

  spblas::sort_state sort_state() const noexcept {
    return matrix_->sort_state();
  }

  // Sets the property in the shared wrapper, for every copy.
  void set_sort_state(spblas::sort_state s) const noexcept {
    matrix_->set_sort_state(s);
  }

  // The allocator given at construction; its unset roles resolve to the
  // backend's default allocator.
  spblas::allocator get_allocator() const noexcept {
    return allocator_;
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

  friend auto tag_invoke(__backend::columns_fn_, matrix_opt matrix)
    requires(__backend::column_iterable<M>)
  {
    return __backend::columns(matrix.base());
  }

  friend auto tag_invoke(__backend::lookup_column_fn_, matrix_opt matrix,
                         index_type column_index)
    requires(__backend::column_lookupable<M>)
  {
    return __backend::lookup_column(matrix.base(), column_index);
  }

  // The wrapper and the backend's optimization data, in one allocation.
  struct shared_data_ {
    explicit shared_data_(M m) : matrix(m) {}

    M matrix;
    __detail::backend_matrix_opt_state_t state;
  };

  spblas::allocator allocator_;
  M* matrix_; // into the shared data, which state_ keeps alive

public:
  // For implementations only: backend-specific optimization data.
  std::shared_ptr<__detail::backend_matrix_opt_state_t> state_;
};

template <typename T>
matrix_opt(T t) -> matrix_opt<T>;

template <typename T>
matrix_opt(T t, spblas::allocator) -> matrix_opt<T>;

// Tells the library that the user changed the *values* of the matrix `m`
// wraps, in place. Only values may change: the sparsity pattern (row/column
// pointers and indices) must stay as it is -- for a new pattern, create a new
// wrapper and matrix_opt. Values inside a matrix_opt may only be changed this
// way, or by library functions such as scale(): the library's optimization
// data for the matrix is derived from them and is dropped here, to be rebuilt
// on next use. Applies to every copy of `m`, which share that data. Operations
// whose later stages rely on the pattern (e.g. multiply_numeric) remain valid.
template <typename M>
void update_matrix(matrix_opt<M>& m) {
  m.state_->invalidate();
}

} // namespace spblas
