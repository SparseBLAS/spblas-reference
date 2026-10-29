#pragma once

#include "mkl.h"
#include <cstdint>

#ifdef MKL_ILP64
// 64-bit indices are to be supported via the mkl_sparse_*_64 entry points,
// not via the ILP64 interface.
#error "oneMKL IE Sparse BLAS backend requires LP64 (MKL_INT = int)"
#endif

namespace spblas {

using index_t = MKL_INT;
using offset_t = index_t;

} // namespace spblas
