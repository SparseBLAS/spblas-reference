#pragma once

#include <oneapi/mkl.hpp>

#include <spblas/detail/log.hpp>
#include <spblas/detail/operation_info_t.hpp>
#include <spblas/detail/ranges.hpp>
#include <spblas/detail/view_inspectors.hpp>

#include <spblas/detail/triangular_types.hpp>

namespace spblas {

//  Mappings from Triangular Solve input args to oneMKL vendor input args
//
//  using   A = L + D + U as a strict decomposition of triangular parts
//
//  spblas_ref input            ->   oneMKL SpTRSV input
//  uplo(op(A))                 ->   op(uplo(A))
//
//  upper + nontrans  (D+U)     ->   nontrans  + upper (D+U)
//  lower + nontrans  (L+D)     ->   nontrans  + lower (L+D)
//  upper + trans     (L+D)^T   ->   trans     + lower (L+D)^T
//  lower + trans     (D+U)^T   ->   trans     + upper (D+U)^T
//  upper + conjtrans (L+D)^H   ->   conjtrans + lower (L+D)^H
//  lower + conjtrans (D+U)^H   ->   conjtrans + upper (D+U)^H
//

template <matrix A, class Triangle, class DiagonalStorage, vector B, vector X>
  requires __detail::has_csr_base<A> &&
           __detail::has_contiguous_range_base<B> &&
           __ranges::contiguous_range<X>
void triangular_solve(A&& a, Triangle uplo, DiagonalStorage diag, B&& b, X&& x)
{
    log_trace("");
    static_assert(std::is_same_v<Triangle, upper_triangle_t> || 
            std::is_same_v<Triangle, lower_triangle_t>);
    static_assert(std::is_same_v<DiagonalStorage, explicit_diagonal_t> || 
            std::is_same_v<DiagonalStorage, implicit_unit_diagonal_t>);

    auto a_base = __detail::get_ultimate_base(a);

    using T = tensor_scalar_t<A>;
    using I = tensor_index_t<A>;
    using O = tensor_offset_t<A>;

    auto alpha_optional = __detail::get_scaling_factor(a, b);
    T alpha = alpha_optional.value_or(1);

    sycl::queue q(sycl::cpu_selector_v);

    oneapi::mkl::sparse::matrix_handle_t a_handle = nullptr;
    oneapi::mkl::sparse::init_matrix_handle(&a_handle);

    oneapi::mkl::sparse::set_csr_data(
            q, a_handle, __backend::shape(a_base)[0], __backend::shape(a_base)[1],
            oneapi::mkl::index_base::zero, a_base.rowptr().data(),
            a_base.colind().data(), a_base.values().data())
        .wait();

    auto op = oneapi::mkl::transpose::nontrans;

    auto uplo_val = std::is_same_v<Triangle, upper_triangle_t> ? 
        oneapi::mkl::uplo::upper : oneapi::mkl::uplo::lower; // someday apply mapping with op

    auto diag_val = std::is_same_v<DiagonalStorage, explicit_diagonal_t> ? 
        oneapi::mkl::diag::nonunit : oneapi::mkl::diag::unit;

    oneapi::mkl::sparse::trsv(q, uplo_val, op, diag_val, alpha, a_handle,
            __ranges::data(b), __ranges::data(x))
        .wait();

    oneapi::mkl::sparse::release_matrix_handle(q, &a_handle).wait();

} // triangular_solve

} // namespace spblas
