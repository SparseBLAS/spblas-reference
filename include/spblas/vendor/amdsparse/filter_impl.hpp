#pragma once

#include <memory>
#include <type_traits>

#include <hip/hip_runtime.h>
#include <hipsparse/hipsparse.h>

#include <amd_spblas/filter.hpp>
#include <spblas/backend/concepts.hpp>
#include <spblas/detail/operation_info_t.hpp>
#include <spblas/detail/ranges.hpp>
#include <spblas/detail/view_inspectors.hpp>

#include "operation_state_t.hpp"
#include "types.hpp"

namespace spblas {

class filter_handle_t {
public:
  filter_handle_t(std::shared_ptr<const allocator> alloc) : alloc_(alloc) {}

  auto result_nnz() {
    return result_nnz_;
  }

  template <matrix A, matrix B, typename Filter>
    requires __detail::has_csr_base<A> && __detail::is_csr_view_v<B>
  void filter_compute(A&& a, B&& b, Filter filter) {

    auto a_base = __detail::get_ultimate_base(a);
    auto b_base = __detail::get_ultimate_base(b);
    using matrix_type = decltype(a_base);
    using input_type = decltype(b_base);
    using value_type = typename matrix_type::scalar_type;

    amd::csr_filter_count_row_nnz(
        __backend::shape(a_base)[0], a_base.values().size(),
        a_base.rowptr().data(), a_base.colind().data(), a_base.values().data(),
        filter, b_base.rowptr().data(), &result_nnz_);
  }

  template <matrix A, matrix B, typename Filter>
    requires __detail::has_csr_base<A> && __detail::is_csr_view_v<B>
  void filter_execute(A&& a, B&& b, Filter filter) {

    auto a_base = __detail::get_ultimate_base(a);
    auto b_base = __detail::get_ultimate_base(b);
    using matrix_type = decltype(a_base);
    using input_type = decltype(b_base);
    using value_type = typename matrix_type::scalar_type;

    amd::csr_filter_fill(__backend::shape(a_base)[0], a_base.values().size(),
                          a_base.rowptr().data(), a_base.colind().data(),
                          a_base.values().data(), filter,
                          b_base.values().size(), b_base.rowptr().data(),
                          b_base.colind().data(), b_base.values().data());
  }

private:
  std::int64_t result_nnz_;
  std::shared_ptr<const allocator> alloc_;
};

template <matrix A, matrix B, typename Filter>
  requires __detail::has_csr_base<A> && __detail::is_csr_view_v<B>
void filter_inspect(filter_handle_t& filter_handle, A&& a, B&& b,
                    Filter filter) {}

template <matrix A, matrix B, typename Filter>
  requires __detail::has_csr_base<A> && __detail::is_csr_view_v<B>
void filter_compute(filter_handle_t& filter_handle, A&& a, B&& b,
                    Filter filter) {
  filter_handle.filter_compute(a, b, filter);
}

template <matrix A, matrix B, typename Filter>
  requires __detail::has_csr_base<A> && __detail::is_csr_view_v<B>
void filter_execute(filter_handle_t& filter_handle, A&& a, B&& b,
                    Filter filter) {
  filter_handle.filter_execute(a, b, filter);
}

} // namespace spblas
