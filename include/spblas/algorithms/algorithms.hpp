#pragma once

#include <spblas/algorithms/scale.hpp>
#include <spblas/algorithms/scale_impl.hpp>

#ifndef SPBLAS_VENDOR_BACKEND
#include <spblas/algorithms/multiply_impl.hpp>
#include <spblas/algorithms/triangular_solve_impl.hpp>
#endif

#include <spblas/algorithms/add.hpp>
#include <spblas/algorithms/add_impl.hpp>

#include <spblas/algorithms/norm_impl.hpp>

// Reference SpMV / SpMM with structure views, in every build (vendors
// override what their library supports).
#include <spblas/algorithms/spmm_structured_impl.hpp>
#include <spblas/algorithms/spmv_structured_impl.hpp>

#include <spblas/algorithms/sort_impl.hpp>

#include <spblas/algorithms/scaled.hpp>
#include <spblas/algorithms/scaled_impl.hpp>

#include <spblas/algorithms/conjugated.hpp>
#include <spblas/algorithms/conjugated_impl.hpp>

#include <spblas/algorithms/transpose.hpp>
#include <spblas/algorithms/transpose_impl.hpp>
