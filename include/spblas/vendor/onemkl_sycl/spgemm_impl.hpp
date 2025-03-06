#pragma once

#include <oneapi/mkl.hpp>

#include <spblas/detail/log.hpp>

#include <spblas/algorithms/transposed.hpp>
#include <spblas/detail/operation_info_t.hpp>
#include <spblas/detail/ranges.hpp>
#include <spblas/detail/view_inspectors.hpp>
#include <spblas/vendor/onemkl_sycl/detail/create_matrix_handle.hpp>

//
// Defines the following APIs for SpGEMM:
//
//  C = op(A) * op(B)
//
//  where A,B and C are sparse matrices of CSR format
//
// operation_info_t multiply_inspect(A, B, C)
// void multiply_compute(operation_info_t, A, B, C)
//

namespace spblas {

template <matrix A, matrix B, matrix C>
  requires(__detail::has_csr_base<A> || __detail::has_csc_base<A>) &&
          (__detail::has_csr_base<B> || __detail::has_csc_base<B>) &&
          __detail::is_csr_view_v<C>
operation_info_t multiply_compute(A&& a, B&& b, C&& c) {
  log_trace("");
  auto a_base = __detail::get_ultimate_base(a);
  auto b_base = __detail::get_ultimate_base(b);

  using oneapi::mkl::transpose;
  using oneapi::mkl::sparse::matmat_request;
  using oneapi::mkl::sparse::matrix_view_descr;

  sycl::queue q(sycl::cpu_selector_v);

  using T = tensor_scalar_t<C>;
  using I = tensor_index_t<C>;

  oneapi::mkl::sparse::matrix_handle_t a_handle =
      __mkl::create_matrix_handle(q, a_base);
  oneapi::mkl::sparse::matrix_handle_t b_handle =
      __mkl::create_matrix_handle(q, b_base);

  I* c_rowptr;
  if (c.rowptr().size() >= __backend::shape(c)[0] + 1) {
    c_rowptr = c.rowptr().data();
  } else {
    c_rowptr = sycl::malloc_device<I>(__backend::shape(c)[0] + 1, q);
  }

  oneapi::mkl::sparse::matrix_handle_t c_handle = nullptr;
  oneapi::mkl::sparse::init_matrix_handle(&c_handle);

  oneapi::mkl::sparse::set_csr_data(
      q, c_handle, __backend::shape(c)[0], __backend::shape(c)[1],
      oneapi::mkl::index_base::zero, c_rowptr, (I*) nullptr, (T*) nullptr)
      .wait();

  oneapi::mkl::sparse::matmat_descr_t descr = nullptr;
  oneapi::mkl::sparse::init_matmat_descr(&descr);

  oneapi::mkl::sparse::set_matmat_data(
      descr, matrix_view_descr::general,
      __mkl::get_transpose(a),                             // view/op for A
      matrix_view_descr::general, __mkl::get_transpose(b), // view/op for B
      matrix_view_descr::general);                         // view for C

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

  index_t nnz = 0;
  if (*c_nnz <= std::numeric_limits<offset_t>::max()) {
    nnz = static_cast<offset_t>(*c_nnz);
  } else {
    std::cout << "Error: c_nnz value out of range for index_t" << std::endl;
    nnz = -1;
  }

  log_info("computed c_nnz = %d", nnz);
  sycl::free(c_nnz, q);

  return operation_info_t{
      index<>{__backend::shape(c)[0], __backend::shape(c)[1]}, nnz,
      __mkl::operation_state_t{a_handle, b_handle, c_handle, nullptr, descr,
                               (void*) c_rowptr, q}};
}

template <matrix A, matrix B, matrix C>
  requires(__detail::has_csr_base<A> || __detail::has_csc_base<A>) &&
          (__detail::has_csr_base<B> || __detail::has_csc_base<B>) &&
          __detail::is_csr_view_v<C>
void multiply_fill(operation_info_t& info, A&& a, B&& b, C&& c) {

  log_trace("");
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

template <matrix A, matrix B, matrix C>
  requires __detail::has_csc_base<A> && __detail::has_csc_base<B> &&
           __detail::is_csc_view_v<C>
operation_info_t multiply_compute(A&& a, B&& b, C&& c) {
  return multiply_compute(transposed(b), transposed(a), transposed(c));
}

template <matrix A, matrix B, matrix C>
  requires((__detail::has_csr_base<A> || __detail::has_csc_base<A>) &&
           (__detail::has_csr_base<B> || __detail::has_csc_base<B>) &&
           __detail::is_csc_view_v<C>)
void multiply_fill(operation_info_t& info, A&& a, B&& b, C&& c) {
  multiply_fill(info, transposed(b), transposed(a), transposed(c));
}

} // namespace spblas
