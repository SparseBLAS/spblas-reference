#pragma once

#include <spblas/vendor/armpl/detail/armpl.hpp>

#include <spblas/detail/operation_info_t.hpp>
#include <spblas/detail/ranges.hpp>
#include <spblas/detail/view_inspectors.hpp>

namespace spblas {

template <matrix A, vector B, vector C>
  requires __detail::has_csr_base<A> &&
           __detail::has_contiguous_range_base<B> &&
           __ranges::contiguous_range<C>
void multiply(A&& a, B&& b, C&& c) {
  auto a_base = __detail::get_ultimate_base(a);
  auto b_base = __detail::get_ultimate_base(b);

  auto alpha_optional = __detail::get_scaling_factor(a, b);
  tensor_scalar_t<A> alpha = alpha_optional.value_or(1);

  armpl_spmat_t a_handle;

  __armpl::create_spmat_csr<tensor_scalar_t<A>>(
      &a_handle, __backend::shape(a_base)[0], __backend::shape(a_base)[1],
      a_base.rowptr().data(), a_base.colind().data(), a_base.values().data(),
      ARMPL_SPARSE_CREATE_NOCOPY);

  auto stat = __armpl::spmv_exec<tensor_scalar_t<A>>(
      ARMPL_SPARSE_OPERATION_NOTRANS, alpha, a_handle, __ranges::data(b_base),
      0, __ranges::data(c));

  armpl_spmat_destroy(a_handle);
}

template <matrix A, matrix B, matrix C>
  requires __detail::has_csr_base<A> && __detail::has_mdspan_matrix_base<B> &&
           __detail::is_matrix_instantiation_of_mdspan_v<C> &&
           std::is_same_v<
               typename __detail::ultimate_base_type_t<B>::layout_type,
               __mdspan::layout_right> &&
           std::is_same_v<typename std::remove_cvref_t<C>::layout_type,
                          __mdspan::layout_right>
void multiply(A&& a, B&& b, C&& c) {
  auto a_base = __detail::get_ultimate_base(a);
  auto b_base = __detail::get_ultimate_base(b);

  auto alpha_optional = __detail::get_scaling_factor(a, b);
  tensor_scalar_t<A> alpha = alpha_optional.value_or(1);

  armpl_spmat_t a_handle, b_handle, c_handle;

  __armpl::create_spmat_csr<tensor_scalar_t<A>>(
      &a_handle, __backend::shape(a_base)[0], __backend::shape(a_base)[1],
      a_base.rowptr().data(), a_base.colind().data(), a_base.values().data(),
      ARMPL_SPARSE_CREATE_NOCOPY);

  __armpl::create_spmat_dense<tensor_scalar_t<B>>(
      &b_handle, ARMPL_ROW_MAJOR, __backend::shape(b_base)[0],
      __backend::shape(b_base)[1], __backend::shape(b_base)[1],
      b_base.data_handle(), ARMPL_SPARSE_CREATE_NOCOPY);

  __armpl::create_spmat_dense<tensor_scalar_t<C>>(
      &c_handle, ARMPL_ROW_MAJOR, __backend::shape(c)[0],
      __backend::shape(c)[1], __backend::shape(c)[1], c.data_handle(),
      ARMPL_SPARSE_CREATE_NOCOPY);

  __armpl::spmm_exec<tensor_scalar_t<A>>(ARMPL_SPARSE_OPERATION_NOTRANS,
                                         ARMPL_SPARSE_OPERATION_NOTRANS, alpha,
                                         a_handle, b_handle, 0, c_handle);

  armpl_int_t m, n;
  tensor_scalar_t<C>* armpl_values;
  __armpl::export_spmat_dense<tensor_scalar_t<C>>(c_handle, ARMPL_ROW_MAJOR, &m,
                                                  &n, &armpl_values);

  std::copy(armpl_values, armpl_values + (m * n), c.data_handle());

  free(armpl_values);

  armpl_spmat_destroy(a_handle);
  armpl_spmat_destroy(b_handle);
  armpl_spmat_destroy(c_handle);
}

template <matrix A, matrix B, matrix C>
  requires __detail::has_csr_base<A> && __detail::has_csr_base<B> &&
           __detail::is_csr_view_v<C>
operation_info_t multiply_inspect(A&& a, B&& b, C&& c) {
  auto a_base = __detail::get_ultimate_base(a);
  auto b_base = __detail::get_ultimate_base(b);

  auto alpha_optional = __detail::get_scaling_factor(a, b);
  tensor_scalar_t<A> alpha = alpha_optional.value_or(1);

  armpl_spmat_t a_handle, b_handle, c_handle;

  __armpl::create_spmat_csr<tensor_scalar_t<A>>(
      &a_handle, __backend::shape(a_base)[0], __backend::shape(a_base)[1],
      a_base.rowptr().data(), a_base.colind().data(), a_base.values().data(),
      ARMPL_SPARSE_CREATE_NOCOPY);

  __armpl::create_spmat_csr<tensor_scalar_t<B>>(
      &b_handle, __backend::shape(b_base)[0], __backend::shape(b_base)[1],
      b_base.rowptr().data(), b_base.colind().data(), a_base.values().data(),
      ARMPL_SPARSE_CREATE_NOCOPY);

  c_handle =
      armpl_spmat_create_null(__backend::shape(c)[0], __backend::shape(c)[1]);

#if 0
/*
  Turns out, there is a problem with my suggestion here.
  If either of the structure hints are set
  (ARMPL_SPARSE_SPMM_STRAT_OPT_PART_STRUCT is used below), then
  `armpl_spmm_optimize` will only have computed any of the
  structure *if the rows of B are sorted*. Otherwise, we return early from
  optimization and execution does everything. Since these are hints,
  this is OK in our implementation. However, it means we can't rely on
  calling `optimize` to *always* calculate NNZ for C in this `inspect`
  function. The only way to guarantee NNZ is to execute.

  If we do the sorting in here then we'll need to sort b_colind, and call
  armpl::create_spmat_csr without ARMPL_SPARSE_CREATE_NOCOPY (flag=0 instead)
  in which case b_colind_copy is copied again, along with taking copies of
  b_rowptr and b_values. Seems excessive!
*/

  auto alpha_hint = alpha == 0 ? ARMPL_SPARSE_SCALAR_ZERO :
                    alpha == 1 ? ARMPL_SPARSE_SCALAR_ONE :
                                 ARMPL_SPARSE_SCALAR_ANY;

  auto beta_hint = ARMPL_SPARSE_SCALAR_ZERO;

  armpl_spmat_hint(c_handle, ARMPL_SPARSE_HINT_SPMM_STRATEGY,
                   ARMPL_SPARSE_SPMM_STRAT_OPT_PART_STRUCT);

  armpl_spmm_optimize(ARMPL_SPARSE_OPERATION_NOTRANS,
                      ARMPL_SPARSE_OPERATION_NOTRANS,
                      ARMPL_SPARSE_SCALAR_ONE, a_handle,
                      b_handle, ARMPL_SPARSE_SCALAR_ZERO, c_handle);

#else

  __armpl::spmm_exec<tensor_scalar_t<A>>(ARMPL_SPARSE_OPERATION_NOTRANS,
                                         ARMPL_SPARSE_OPERATION_NOTRANS, alpha,
                                         a_handle, b_handle, 0, c_handle);
#endif

  armpl_int_t index_base, m, n, nnz;
  armpl_spmat_query(c_handle, &index_base, &m, &n, &nnz);

  return operation_info_t(
      index<>{__backend::shape(c)[0], __backend::shape(c)[1]}, nnz,
      __armpl::operation_state_t{a_handle, b_handle, c_handle, nullptr});
}

template <matrix A, matrix B, matrix C>
  requires __detail::has_csr_base<A> && __detail::has_csr_base<B> &&
           __detail::is_csr_view_v<C>
void multiply_execute(operation_info_t& info, A&& a, B&& b, C&& c) {
  auto a_handle = info.state_.a_handle;
  auto b_handle = info.state_.b_handle;
  auto c_handle = info.state_.c_handle;

  armpl_int_t m, n;
  auto nnz = info.result_nnz();
  armpl_int_t *rowptr, *colind;
  tensor_scalar_t<C>* values;

#if 0
  auto alpha_optional = __detail::get_scaling_factor(a, b);
  tensor_scalar_t<A> alpha = alpha_optional.value_or(1);

  // We would do this here instead if B's rows were guaranteed to be sorted
  __armpl::spmm_exec<tensor_scalar_t<A>>(ARMPL_SPARSE_OPERATION_NOTRANS,
                                         ARMPL_SPARSE_OPERATION_NOTRANS, alpha,
                                         a_handle, b_handle, 0, c_handle);
#endif

  __armpl::export_spmat_csr<tensor_scalar_t<C>>(c_handle, 0, &m, &n, &rowptr,
                                                &colind, &values);

  std::copy(values, values + nnz, c.values().begin());
  std::copy(colind, colind + nnz, c.colind().begin());
  std::copy(rowptr, rowptr + m + 1, c.rowptr().begin());

  free(values);
  free(rowptr);
  free(colind);
}

} // namespace spblas
