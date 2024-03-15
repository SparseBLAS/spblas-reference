#pragma once

#include <spblas/algorithms/scale.hpp>

#ifndef ENABLE_ONEMKL
#include <spblas/algorithms/scale_impl.hpp>
#endif

#include <spblas/algorithms/multiply.hpp>

#ifndef ENABLE_ONEMKL
#include <spblas/algorithms/multiply_impl.hpp>
#endif

#include <spblas/algorithms/scaled.hpp>
#include <spblas/algorithms/scaled_impl.hpp>
