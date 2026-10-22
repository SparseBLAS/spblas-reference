#pragma once

//
// Reference multiply, one file per operation:
//
//   SpMV    spmv_impl.hpp    sparse A, dense vectors x, y [, z]
//   SpMM    spmm_impl.hpp    sparse A, dense matrices X, Y [, Z]
//   SpGEMM  spgemm_impl.hpp  sparse A, B, C (kernels in detail/spgemm/)
//
// Each file lists the formats and views it supports. Variants for structured
// views of A (view::triangle / symmetric / hermitian) belong in their own
// files next to these, e.g. spmv_triangle_impl.hpp.
//

#include <spblas/algorithms/spgemm_impl.hpp>
#include <spblas/algorithms/spmm_impl.hpp>
#include <spblas/algorithms/spmm_structured_impl.hpp>
#include <spblas/algorithms/spmv_impl.hpp>
#include <spblas/algorithms/spmv_structured_impl.hpp>

// The policy-first overloads (reference: the policy is ignored).
#include <spblas/algorithms/policy_forwarding.hpp>
