#pragma once

#include <spblas/concepts.hpp>
#include <spblas/detail/operation_info_t.hpp>

namespace spblas {

// SpMV variants
template <matrix A, vector B, vector C>
operation_info_t multiply_inspect(A&& a, B&& b, C&& c);

template <matrix A, vector B, vector C>
void multiply_inspect(operation_info_t& info, A&& a, B&& b, C&& c);

template <matrix A, vector B, vector C>
void multiply(A&& a, B&& b, C&& c);

template <matrix A, vector B, vector C>
void multiply(operation_into_t& info, A&& a, B&& b, C&& c);


// SpMM variants
template <matrix A, matrix B, matrix C>
void multiply(A&& a, B&& b, C&& c);

template <matrix A, matrix B, matrix C>
void multiply(operation_info_t& info, A&& a, B&& b, C&& c);

// SpMM and SpGEMM multiply_inspect variants
template <matrix A, matrix B, matrix C>
operation_info_t multiply_inspect(A&& a, B&& b, C&& c);

template <matrix A, matrix B, matrix C>
void multiply_inspect(operation_info_t& info, A&& a, B&& b, C&& c);


// SpGEMM variants
template <typename ExecutionPolicy, matrix A, matrix B, matrix C>
operation_info_t multiply_compute(ExecutionPolicy &&policy, A&& a, B&& b, C&& c);

template <typename ExecutionPolicy, matrix A, matrix B, matrix C>
void multiply_compute(ExecutionPolicy &&policy, operation_info_t& info, A&& a, B&& b, C&& c);

template <typename ExecutionPolicy, matrix A, matrix B, matrix C>
void multiply_fill(ExecutionPolicy &&policy, operation_info_t& info, A&& a, B&& b, C&& c);

template <matrix A, matrix B, matrix C>
operation_info_t multiply_compute(A&& a, B&& b, C&& c);

template <matrix A, matrix B, matrix C>
void multiply_compute(operation_info_t& info, A&& a, B&& b, C&& c);

template <matrix A, matrix B, matrix C>
void multiply_fill(operation_info_t& info, A&& a, B&& b, C&& c);

} // namespace spblas
