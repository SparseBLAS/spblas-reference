#pragma once

#include <spblas/backend/algorithms.hpp>
#include <spblas/backend/concepts.hpp>
#include <spblas/backend/cpos.hpp>
#include <spblas/backend/generate.hpp>
#include <spblas/backend/view_customizations.hpp>

#ifdef SPBLAS_ENABLE_ONEMKL_SYCL
#include <spblas/vendor/onemkl_sycl/onemkl_sycl.hpp>
#endif

#ifdef SPBLAS_ENABLE_ARMPL
#include <spblas/vendor/armpl/armpl.hpp>
#endif
