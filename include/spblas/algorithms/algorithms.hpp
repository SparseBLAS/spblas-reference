#pragma once

#include <spblas/algorithms/scale.hpp>

#include <spblas/algorithms/multiply.hpp>

#include <spblas/algorithms/scaled.hpp>
#ifdef NVIDIA
#include <spblas/algorithms/nvidia/multiply_impl.hpp>
#include <spblas/algorithms/nvidia/scale_impl.hpp>
#include <spblas/algorithms/nvidia/scaled_impl.hpp>
#else
#include <spblas/algorithms/multiply_impl.hpp>
#include <spblas/algorithms/scale_impl.hpp>
#include <spblas/algorithms/scaled_impl.hpp>
#endif
