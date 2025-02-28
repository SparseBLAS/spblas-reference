#pragma once

#if defined(SPBLAS_ENABLE_ONEMKL_SYCL) || defined(SPBLAS_ENABLE_ARMPL) ||      \
    defined(SPBLAS_ENABLE_AOCLSPARSE) || defined(SPBLAS_ENABLE_ROCSPARSE) ||   \
    defined(SPBLAS_ENABLE_CUSPARSE)
#define SPBLAS_VENDOR_BACKEND true
#endif

#include <spblas/algorithms/algorithms.hpp>
#include <spblas/concepts.hpp>
#include <spblas/views/views.hpp>

#include <spblas/backend/backend.hpp>
