#pragma once

//
// oneMKL Inspector-Executor Sparse BLAS backend, one file per operation:
//
//   SpMV              spmv_impl.hpp              mkl_sparse_?_mv
//   SpMM              spmm_impl.hpp              mkl_sparse_?_mm
//   SpGEMM            spgemm_impl.hpp            mkl_sparse_sp2m
//   triangular solve  triangular_solve_impl.hpp  mkl_sparse_?_trsv
//   SpADD             add_impl.hpp               mkl_sparse_?_add where it
//                                                applies, else reference
//   sort_matrix       sort_impl.hpp              mkl_sparse_order
//
// How operands map onto MKL handles and operations: detail/operand.hpp. The
// C wrappers: mkl_wrappers.hpp. transpose, scale and the norms use the
// reference implementations, which every backend includes.
//

#include "add_impl.hpp"
#include "sort_impl.hpp"
#include "spgemm_impl.hpp"
#include "spmm_impl.hpp"
#include "spmm_structured_impl.hpp"
#include "spmv_impl.hpp"
#include "spmv_structured_impl.hpp"
#include "triangular_solve_impl.hpp"

// The policy-first overloads (the policy is ignored).
#include <spblas/algorithms/policy_forwarding.hpp>
