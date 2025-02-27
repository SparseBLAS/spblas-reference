#pragma once

#include <spblas/vendor/armpl/detail/armpl.hpp>

#include <spblas/detail/log.hpp>
#include <spblas/detail/operation_info_t.hpp>
#include <spblas/detail/ranges.hpp>
#include <spblas/detail/view_inspectors.hpp>

#include <spblas/detail/triangular_types.hpp>

namespace spblas {

template <matrix A, class Triangle, class DiagonalStorage, vector B, vector X>
  requires __detail::has_csr_base<A> &&
           __detail::has_contiguous_range_base<B> &&
           __ranges::contiguous_range<X>
void triangular_solve(A&& a, Triangle uplo, DiagonalStorage diag, B&& b,
                      X&& x) {
  log_trace("");
  static_assert(std::is_same_v<Triangle, upper_triangle_t> ||
                std::is_same_v<Triangle, lower_triangle_t>);
  static_assert(std::is_same_v<DiagonalStorage, explicit_diagonal_t> ||
                std::is_same_v<DiagonalStorage, implicit_unit_diagonal_t>);

  auto a_base = __detail::get_ultimate_base(a);
  auto b_base = __detail::get_ultimate_base(b);

  using T = tensor_scalar_t<A>;
  using I = tensor_index_t<A>;
  using O = tensor_offset_t<A>;

  auto m = __backend::shape(a_base)[0];
  auto n = __backend::shape(a_base)[1];

  auto alpha_optional = __detail::get_scaling_factor(a, b);
  T alpha = alpha_optional.value_or(1);

  armpl_spmat_t a_handle;

  // Optimistically try the solve without a copy, in case the matrix is already
  // triangular
  __armpl::create_spmat_csr<tensor_scalar_t<A>>(
      &a_handle, m, n, a_base.rowptr().data(), a_base.colind().data(),
      a_base.values().data(), ARMPL_SPARSE_CREATE_NOCOPY);

  auto stat = __armpl::sptrsv_exec<tensor_scalar_t<A>>(
      ARMPL_SPARSE_OPERATION_NOTRANS, a_handle, __ranges::data(x), alpha,
      __ranges::data(b_base));

  armpl_spmat_destroy(a_handle);

  if (stat != ARMPL_STATUS_SUCCESS) {

    //  Arm PL needs a copy of the matrix corresponding to the specified
    //  triangule with the diagonal set appropriately.

    auto is_upper = std::is_same_v<Triangle, upper_triangle_t>;
    auto is_unit = std::is_same_v<DiagonalStorage, implicit_unit_diagonal_t>;

    auto colind = a_base.colind().data();
    auto rowptr = a_base.rowptr().data();
    auto values = a_base.values().data();

    std::vector<T> tmp_values;
    std::vector<I> tmp_rowptr(m + 1);
    std::vector<O> tmp_colind;

    auto index_base = rowptr[0];

    auto is_included = [&](auto r, auto c) {
      if (is_unit) {
        if (is_upper) {
          return r < c;
        } else {
          return r > c;
        }
      } else {
        if (is_upper) {
          return r <= c;
        } else {
          return r >= c;
        }
      }
    };

    int k = 0;
    for (armpl_int_t r = 0; r < m; r++) {

      if (is_unit && is_upper) {
        tmp_colind.push_back(r);
        tmp_values.push_back(T(1));
        k++;
      }

      for (auto i = rowptr[r] - index_base; i < rowptr[r + 1] - index_base;
           i++) {
        auto c = colind[i];
        auto v = values[i];

        if (is_included(r, c)) {
          tmp_colind.push_back(c);
          tmp_values.push_back(v);
          k++;
        }
      }

      if (is_unit && !is_upper) {
        tmp_colind.push_back(r);
        tmp_values.push_back(T(1));
        k++;
      }

      tmp_rowptr[r + 1] = k;
    }

    __armpl::create_spmat_csr<tensor_scalar_t<A>>(
        &a_handle, m, n, tmp_rowptr.data(), tmp_colind.data(),
        tmp_values.data(), ARMPL_SPARSE_CREATE_NOCOPY);

    stat = __armpl::sptrsv_exec<tensor_scalar_t<A>>(
        ARMPL_SPARSE_OPERATION_NOTRANS, a_handle, __ranges::data(x), alpha,
        __ranges::data(b_base));
    if (stat != ARMPL_STATUS_SUCCESS) {
      armpl_spmat_print_err(a_handle);
      assert(false);
    }

    armpl_spmat_destroy(a_handle);
  }

} // triangular_solve

} // namespace spblas
