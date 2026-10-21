#pragma once

#include <memory>
#include <memory_resource>

#include <spblas/backend/concepts.hpp>
#include <spblas/backend/cpos.hpp>
#include <spblas/concepts.hpp>
#include <spblas/detail/backend_state.hpp>

namespace spblas {

// A matrix wrapper plus opaque, library-owned optimization data for it.
// Copies share the optimization data, which is released with the last copy.
// The data is allocated from the memory resource given at construction, or
// the backend's default resource if none is given.
template <matrix M>
  requires(__detail::view<M> &&
           (__detail::is_csr_view_v<M> || __detail::is_csc_view_v<M>) )
class matrix_opt<M> : public view_base {
public:
  using scalar_type = tensor_scalar_t<M>;
  using scalar_reference = tensor_scalar_reference_t<M>;
  using index_type = tensor_index_t<M>;
  using offset_type = tensor_offset_t<M>;

  explicit matrix_opt(M matrix, std::pmr::memory_resource* resource = nullptr)
      : matrix_(matrix),
        resource_(resource != nullptr ? resource
                                      : __detail::default_memory_resource()),
        state_(std::allocate_shared<__detail::backend_matrix_opt_state_t>(
            std::pmr::polymorphic_allocator<
                __detail::backend_matrix_opt_state_t>(resource_))) {}

  matrix_opt(M matrix, const std::pmr::polymorphic_allocator<>& allocator)
      : matrix_opt(matrix, allocator.resource()) {}

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

  std::pmr::polymorphic_allocator<> get_allocator() const noexcept {
    return std::pmr::polymorphic_allocator<>(resource_);
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

private:
  std::pmr::memory_resource* resource_;

public:
  // For implementations only: backend-specific optimization data.
  std::shared_ptr<__detail::backend_matrix_opt_state_t> state_;
};

template <typename T>
matrix_opt(T t) -> matrix_opt<T>;

template <typename T>
matrix_opt(T t, std::pmr::memory_resource*) -> matrix_opt<T>;

template <typename T>
matrix_opt(T t, const std::pmr::polymorphic_allocator<>&) -> matrix_opt<T>;

} // namespace spblas
