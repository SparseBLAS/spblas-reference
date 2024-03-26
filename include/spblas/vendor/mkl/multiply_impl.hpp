#pragma once

#include <oneapi/mkl.hpp>

#include <spblas/detail/operation_info_t.hpp>
#include <spblas/detail/ranges.hpp>

#include <fmt/ranges.h>

namespace spblas {

template <matrix A, vector B, vector C>
  requires __detail::is_csr_view_v<A> && __ranges::contiguous_range<B> &&
           __ranges::contiguous_range<C>
void multiply(A&& a, B&& b, C&& c) {
  fmt::print("Running Intel SpMV multiply...\n");
  sycl::queue q(sycl::cpu_selector_v);
  oneapi::mkl::sparse::matrix_handle_t a_handle = nullptr;

  oneapi::mkl::sparse::init_matrix_handle(&a_handle);

  oneapi::mkl::sparse::set_csr_data(
      q, a_handle, __backend::shape(a)[0], __backend::shape(a)[1],
      oneapi::mkl::index_base::zero, a.rowptr().data(), a.colind().data(),
      a.values().data())
      .wait();

  oneapi::mkl::sparse::gemv(q, oneapi::mkl::transpose::nontrans, 1.0, a_handle,
                            __ranges::data(b), 0.0, __ranges::data(c))
      .wait();

  oneapi::mkl::sparse::release_matrix_handle(q, &a_handle).wait();
}

template <matrix A, matrix B, matrix C>
  requires __detail::is_csr_view_v<A> &&
           __detail::is_matrix_instantiation_of_mdspan_v<B> &&
           __detail::is_matrix_instantiation_of_mdspan_v<C> &&
           std::is_same_v<typename std::remove_cvref_t<B>::layout_type,
                          __mdspan::layout_right> &&
           std::is_same_v<typename std::remove_cvref_t<C>::layout_type,
                          __mdspan::layout_right>
void multiply(A&& a, B&& b, C&& c) {
  fmt::print("Running Intel SpMM multiply...\n");
  sycl::queue q(sycl::cpu_selector_v);

  oneapi::mkl::sparse::matrix_handle_t a_handle = nullptr;
  oneapi::mkl::sparse::init_matrix_handle(&a_handle);

  oneapi::mkl::sparse::set_csr_data(
      q, a_handle, __backend::shape(a)[0], __backend::shape(a)[1],
      oneapi::mkl::index_base::zero, a.rowptr().data(), a.colind().data(),
      a.values().data())
      .wait();

  oneapi::mkl::sparse::gemm(
      q, oneapi::mkl::layout::row_major, oneapi::mkl::transpose::nontrans,
      oneapi::mkl::transpose::nontrans, 1.0, a_handle, b.data_handle(),
      b.extent(1), b.extent(1), 0.0, c.data_handle(), c.extent(1))
      .wait();

  oneapi::mkl::sparse::release_matrix_handle(q, &a_handle).wait();
}

template <matrix A, matrix B, matrix C>
  requires __detail::is_csr_view_v<A> && __detail::is_csr_view_v<B> &&
           __detail::is_csr_view_v<C>
operation_info_t multiply_inspect(A&& a, B&& b, C&& c) {
  fmt::print("Running Intel SpGEMM multiply_inspect...\n");
  using oneapi::mkl::transpose;
  using oneapi::mkl::sparse::matmat_request;
  using oneapi::mkl::sparse::matrix_view_descr;

  oneapi::mkl::sparse::matmat_descr_t descr = nullptr;

  sycl::queue q(sycl::cpu_selector_v);

  fmt::print("Init...\n");

  oneapi::mkl::sparse::init_matmat_descr(&descr);

  oneapi::mkl::sparse::set_matmat_data(
      descr, matrix_view_descr::general, transpose::nontrans, // view/op for A
      matrix_view_descr::general, transpose::nontrans,        // view/op for B
      matrix_view_descr::general);                            // view for C

  oneapi::mkl::sparse::matrix_handle_t a_handle, b_handle, c_handle;
  a_handle = b_handle = c_handle = nullptr;

  fmt::print("Initing handles...\n");
  oneapi::mkl::sparse::init_matrix_handle(&a_handle);
  oneapi::mkl::sparse::init_matrix_handle(&b_handle);
  oneapi::mkl::sparse::init_matrix_handle(&c_handle);
  fmt::print("Filling data...\n");

  oneapi::mkl::sparse::set_csr_data(
      q, a_handle, __backend::shape(a)[0], __backend::shape(a)[1],
      oneapi::mkl::index_base::zero, a.rowptr().data(), a.colind().data(),
      a.values().data())
      .wait();

  oneapi::mkl::sparse::set_csr_data(
      q, b_handle, __backend::shape(b)[0], __backend::shape(b)[1],
      oneapi::mkl::index_base::zero, b.rowptr().data(), b.colind().data(),
      b.values().data())
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

  fmt::print("Work estimation...\n");
  auto ev1 = oneapi::mkl::sparse::matmat(q, a_handle, b_handle, c_handle,
                                         matmat_request::work_estimation, descr,
                                         nullptr, nullptr, {});

  fmt::print("Compute...\n");
  auto ev2 = oneapi::mkl::sparse::matmat(q, a_handle, b_handle, c_handle,
                                         matmat_request::compute, descr,
                                         nullptr, nullptr, {ev1});

  fmt::print("Getting C NNZ...\n");
  std::int64_t* c_nnz = sycl::malloc_host<std::int64_t>(1, q);

  auto ev3_1 = oneapi::mkl::sparse::matmat(q, a_handle, b_handle, c_handle,
                                           matmat_request::get_nnz, descr,
                                           c_nnz, nullptr, {ev2});

  ev3_1.wait(); // sync is required to read c_nnz

  fmt::print("Get NNZ...\n");
  std::int64_t nnz = *c_nnz;

  sycl::free(c_nnz, q);

  fmt::print("After free...\n");

  fmt::print("Returning operation_info...\n");
  return operation_info_t{
      index<>{__backend::shape(a)[0], __backend::shape(a)[1]}, nnz,
      __mkl::operation_state_t{a_handle, b_handle, c_handle, nullptr, descr,
                               (void*) c_rowptr, q}};
}

template <matrix A, matrix B, matrix C>
  requires __detail::is_csr_view_v<A> && __detail::is_csr_view_v<B> &&
           __detail::is_csr_view_v<C>
void multiply_execute(operation_info_t& info, A&& a, B&& b, C&& c) {
  fmt::print("Running Intel SpGEMM multiply_execute...\n");
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
}

} // namespace spblas
