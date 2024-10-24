#pragma once

#include <memory>
#include <type_traits>

#include <cuda_runtime.h>
#include <cusparse.h>

#include <cuspblas/csr_add.hpp>
#include <spblas/backend/concepts.hpp>
#include <spblas/detail/operation_info_t.hpp>
#include <spblas/detail/ranges.hpp>
#include <spblas/detail/view_inspectors.hpp>

#include "operation_state_t.hpp"
#include "types.hpp"

namespace spblas {

class add_handle_t {
public:
  add_handle_t(std::shared_ptr<const allocator> alloc) : alloc_(alloc) {}

  auto result_nnz() {
    return result_nnz_;
  }

  template <matrix A, matrix B, matrix C>
    requires __detail::has_csr_base<A> && __detail::has_csr_base<B> &&
             __detail::is_csr_view_v<C>
  void add_compute(A&& a, B&& b, C&& c) {

    auto a_base = __detail::get_ultimate_base(a);
    auto b_base = __detail::get_ultimate_base(b);
    using matrix_type = decltype(a_base);
    using input_type = decltype(b_base);
    using output_type = std::remove_reference_t<decltype(c)>;
    using value_type = typename matrix_type::scalar_type;

    auto alpha_optional = __detail::get_scaling_factor(a, b);
    tensor_scalar_t<A> alpha = alpha_optional.value_or(1);

    value_type alpha_val = alpha;
    value_type beta = 0.0;

    cuda::csr_add_count_row_nnz(__backend::shape(a_base)[0],
                                a_base.values().size(), a_base.rowptr().data(),
                                a_base.colind().data(), b_base.values().size(),
                                b_base.rowptr().data(), b_base.colind().data(),
                                c.rowptr().data(), &result_nnz_);
  }

  template <matrix A, matrix B, matrix C>
    requires __detail::has_csr_base<A> && __detail::has_csr_base<B> &&
             __detail::is_csr_view_v<C>
  void add_execute(A&& a, B&& b, C&& c) {

    auto a_base = __detail::get_ultimate_base(a);
    auto b_base = __detail::get_ultimate_base(b);
    using matrix_type = decltype(a_base);
    using input_type = decltype(b_base);
    using output_type = std::remove_reference_t<decltype(c)>;
    using value_type = typename matrix_type::scalar_type;

    auto alpha_optional = __detail::get_scaling_factor(a, b);
    tensor_scalar_t<A> alpha = alpha_optional.value_or(1);

    value_type alpha_val = alpha;
    value_type beta = 0.0;

    cuda::csr_add_fill(__backend::shape(a_base)[0], a_base.values().size(),
                       a_base.rowptr().data(), a_base.colind().data(),
                       a_base.values().data(), b_base.values().size(),
                       b_base.rowptr().data(), b_base.colind().data(),
                       b_base.values().data(), c.values().size(),
                       c.rowptr().data(), c.colind().data(), c.values().data());
  }

private:
  std::int64_t result_nnz_;
  std::shared_ptr<const allocator> alloc_;
};

template <matrix A, matrix B, matrix C>
  requires __detail::has_csr_base<A> && __detail::has_csr_base<B> &&
           __detail::is_csr_view_v<C>
void add_inspect(add_handle_t& add_handle, A&& a, B&& b, C&& c) {}

template <matrix A, matrix B, matrix C>
  requires __detail::has_csr_base<A> && __detail::has_csr_base<B> &&
           __detail::is_csr_view_v<C>
void add_compute(add_handle_t& add_handle, A&& a, B&& b, C&& c) {
  add_handle.add_compute(a, b, c);
}

template <matrix A, matrix B, matrix C>
  requires __detail::has_csr_base<A> && __detail::has_csr_base<B> &&
           __detail::is_csr_view_v<C>
void add_execute(add_handle_t& add_handle, A&& a, B&& b, C&& c) {
  add_handle.add_execute(a, b, c);
}

} // namespace spblas
