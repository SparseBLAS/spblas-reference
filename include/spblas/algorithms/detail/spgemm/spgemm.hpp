#pragma once

// Reference SpGEMM kernels, C = A * B, in spblas::__detail:
//   spgemm_nnz(a, b, c, allocator)   number of entries in C's symbolic
//                                    pattern
//   spgemm_fill(a, b, c, allocator)  fills C's arrays (sized by spgemm_nnz)
//
//   Their scratch storage (row accumulators etc.) comes from `allocator`, a
//   std::pmr::polymorphic_allocator<>; operations pass their
//   operation_state's.
//
//   A    B    C    kernel
//   CSR  CSR  CSR  Gustavson                    spgemm_gustavsons.hpp
//   CSR  CSR  CSC  Gustavson, scattered         spgemm_gustavsons.hpp
//   CSC  CSC  CSC  Gustavson on the transposes  spgemm_gustavsons.hpp
//   CSR  CSC  CSR  inner product                spgemm_innerproduct.hpp
//   CSR  CSC  CSC  inner product, transposed    spgemm_innerproduct.hpp
//   CSC  CSR  CSR  outer product                spgemm_outerproduct.hpp
//   CSC  CSR  CSC  outer product, transposed    spgemm_outerproduct.hpp
//   CSC  CSC  CSR  not supported
//
// "CSR" means row-iterable and "CSC" column-iterable, so matrix_opt and
// view::conjugate of either work too.

#include "spgemm_gustavsons.hpp"
#include "spgemm_innerproduct.hpp"
#include "spgemm_outerproduct.hpp"
