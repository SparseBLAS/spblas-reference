#pragma once

#if defined(SPBLAS_ENABLE_ONEMKL) || defined(SPBLAS_ENABLE_ARMPL)
#define SPBLAS_VENDOR_BACKEND true
#endif

#include <spblas/algorithms/algorithms.hpp>
#include <spblas/concepts.hpp>
#include <spblas/views/views.hpp>

#include <spblas/backend/backend.hpp>
