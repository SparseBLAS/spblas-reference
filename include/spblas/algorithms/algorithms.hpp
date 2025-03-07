#pragma once

#include <spblas/algorithms/scale.hpp>
#include <spblas/algorithms/scale_impl.hpp>

// #include <spblas/algorithms/multiply.hpp>

#ifndef SPBLAS_VENDOR_BACKEND
#include <spblas/algorithms/multiply_impl.hpp>
#include <spblas/algorithms/triangular_solve_impl.hpp>
#endif

#include <spblas/algorithms/add.hpp>
#include <spblas/algorithms/add_impl.hpp>

#include <spblas/algorithms/scaled.hpp>
#include <spblas/algorithms/scaled_impl.hpp>

#include <spblas/algorithms/transpose.hpp>
#include <spblas/algorithms/transpose_impl.hpp>
