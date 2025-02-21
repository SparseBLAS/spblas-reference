#pragma once

#include <oneapi/mkl.hpp>

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

  sycl::queue q(sycl::cpu_selector_v);
  oneapi::mkl::sparse::matrix_handle_t a_handle = nullptr;

  oneapi::mkl::sparse::init_matrix_handle(&a_handle);

  oneapi::mkl::sparse::set_csr_data(
      q, a_handle, __backend::shape(a_base)[0], __backend::shape(a_base)[1],
      oneapi::mkl::index_base::zero, a_base.rowptr().data(),
      a_base.colind().data(), a_base.values().data())
      .wait();

  oneapi::mkl::sparse::gemv(q, oneapi::mkl::transpose::nontrans, alpha,
                            a_handle, __ranges::data(b_base), 0.0,
                            __ranges::data(c))
      .wait();

  c[0] += 1;

  oneapi::mkl::sparse::release_matrix_handle(q, &a_handle).wait();
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

  sycl::queue q(sycl::cpu_selector_v);

  oneapi::mkl::sparse::matrix_handle_t a_handle = nullptr;
  oneapi::mkl::sparse::init_matrix_handle(&a_handle);

  oneapi::mkl::sparse::set_csr_data(
      q, a_handle, __backend::shape(a_base)[0], __backend::shape(a_base)[1],
      oneapi::mkl::index_base::zero, a_base.rowptr().data(),
      a_base.colind().data(), a_base.values().data())
      .wait();

  oneapi::mkl::sparse::gemm(
      q, oneapi::mkl::layout::row_major, oneapi::mkl::transpose::nontrans,
      oneapi::mkl::transpose::nontrans, alpha, a_handle, b_base.data_handle(),
      b_base.extent(1), b_base.extent(1), 0.0, c.data_handle(), c.extent(1))
      .wait();

  oneapi::mkl::sparse::release_matrix_handle(q, &a_handle).wait();
}

template <matrix A, matrix B, matrix C>
  requires __detail::has_csr_base<A> && __detail::has_csr_base<B> &&
           __detail::is_csr_view_v<C>
operation_info_t multiply_inspect(A&& a, B&& b, C&& c) {
  auto a_base = __detail::get_ultimate_base(a);
  auto b_base = __detail::get_ultimate_base(b);

  using oneapi::mkl::transpose;
  using oneapi::mkl::sparse::matmat_request;
  using oneapi::mkl::sparse::matrix_view_descr;

  oneapi::mkl::sparse::matmat_descr_t descr = nullptr;

  sycl::queue q(sycl::cpu_selector_v);

  oneapi::mkl::sparse::init_matmat_descr(&descr);

  oneapi::mkl::sparse::set_matmat_data(
      descr, matrix_view_descr::general, transpose::nontrans, // view/op for A
      matrix_view_descr::general, transpose::nontrans,        // view/op for B
      matrix_view_descr::general);                            // view for C

  oneapi::mkl::sparse::matrix_handle_t a_handle, b_handle, c_handle;
  a_handle = b_handle = c_handle = nullptr;

  oneapi::mkl::sparse::init_matrix_handle(&a_handle);
  oneapi::mkl::sparse::init_matrix_handle(&b_handle);
  oneapi::mkl::sparse::init_matrix_handle(&c_handle);

  oneapi::mkl::sparse::set_csr_data(
      q, a_handle, __backend::shape(a_base)[0], __backend::shape(a_base)[1],
      oneapi::mkl::index_base::zero, a_base.rowptr().data(),
      a_base.colind().data(), a_base.values().data())
      .wait();

  oneapi::mkl::sparse::set_csr_data(
      q, b_handle, __backend::shape(b_base)[0], __backend::shape(b_base)[1],
      oneapi::mkl::index_base::zero, b_base.rowptr().data(),
      b_base.colind().data(), b_base.values().data())
      .wait();

  using T = tensor_scalar_t<C>;
  using I = tensor_index_t<C>;

  I* c_rowptr;
  if (c.rowptr().size() >= __backend::shape(c)[0] + 1) {
    c_rowptr = c.rowptr().data();
  } else {
    c_rowptr = sycl::malloc_device<I>(__backend::shape(c)[0] + 1, q);
  }

  oneapi::mkl::sparse::set_csr_data(
      q, c_handle, __backend::shape(c)[0], __backend::shape(c)[1],
      oneapi::mkl::index_base::zero, c_rowptr, (I*) nullptr, (T*) nullptr)
      .wait();

  auto ev1 = oneapi::mkl::sparse::matmat(q, a_handle, b_handle, c_handle,
                                         matmat_request::work_estimation, descr,
                                         nullptr, nullptr, {});

  auto ev2 = oneapi::mkl::sparse::matmat(q, a_handle, b_handle, c_handle,
                                         matmat_request::compute, descr,
                                         nullptr, nullptr, {ev1});

  std::int64_t* c_nnz = sycl::malloc_host<std::int64_t>(1, q);

  auto ev3_1 = oneapi::mkl::sparse::matmat(q, a_handle, b_handle, c_handle,
                                           matmat_request::get_nnz, descr,
                                           c_nnz, nullptr, {ev2});

  ev3_1.wait(); // sync is required to read c_nnz

  std::int64_t nnz = *c_nnz;

  sycl::free(c_nnz, q);

  return operation_info_t{
      index<>{__backend::shape(c)[0], __backend::shape(c)[1]}, nnz,
      __mkl::operation_state_t{a_handle, b_handle, c_handle, nullptr, descr,
                               (void*) c_rowptr, q}};
}

template <matrix A, matrix B, matrix C>
  requires __detail::has_csr_base<A> && __detail::has_csr_base<B> &&
           __detail::is_csr_view_v<C>
void multiply_execute(operation_info_t& info, A&& a, B&& b, C&& c) {
  auto a_base = __detail::get_ultimate_base(a);
  auto b_base = __detail::get_ultimate_base(b);

  auto alpha_optional = __detail::get_scaling_factor(a, b);
  tensor_scalar_t<A> alpha = alpha_optional.value_or(1);

  using oneapi::mkl::sparse::matmat_request;
  sycl::queue q(sycl::cpu_selector_v);
  using I = tensor_index_t<C>;

  I* c_rowptr = (I*) info.state_.c_rowptr;

  auto a_handle = info.state_.a_handle;
  auto b_handle = info.state_.b_handle;
  auto c_handle = info.state_.c_handle;

  auto descr = info.state_.descr;

  auto ev_setC = oneapi::mkl::sparse::set_csr_data(
      q, c_handle, __backend::shape(c)[0], __backend::shape(c)[1],
      oneapi::mkl::index_base::zero, c_rowptr, c.colind().data(),
      c.values().data());

  auto e = oneapi::mkl::sparse::matmat(q, a_handle, b_handle, c_handle,
                                       matmat_request::finalize, descr, nullptr,
                                       nullptr, {ev_setC});

  e.wait();

  if (c_rowptr != c.rowptr().data()) {
    q.memcpy(c.rowptr().data(), c_rowptr,
             sizeof(I) * (__backend::shape(c)[0] + 1))
        .wait();

    sycl::free(c_rowptr, q);
  }

  if (alpha_optional.has_value()) {
    scale(alpha, c);
  }
}

} // namespace spblas
