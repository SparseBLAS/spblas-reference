#pragma once

#ifdef SPBLAS_ENABLE_ONEMKL
#define SPBLAS_VENDOR_BACKEND true
#endif

#include <spblas/algorithms/algorithms.hpp>
#include <spblas/concepts.hpp>
#include <spblas/views/views.hpp>

#include <spblas/backend/backend.hpp>
