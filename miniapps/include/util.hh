#ifndef miniapps_util_h
#define miniapps_util_h

#include "rl_blaspp.hh"
#include "rl_lapackpp.hh"
#include <spblas/spblas.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>

namespace miniapps::util {

/// A version of the above function to be used on a vector of integers
template <typename T>
void col_swap(int64_t m, int64_t n, int64_t k, T* A, int64_t lda,
              std::vector<int64_t> idx) {
  if (k > n)
    throw std::runtime_error("Invalid rank parameter.");

  int64_t i, j; //, l;
  for (i = 0, j = 0; i < k; ++i) {
    j = idx[i] - 1;
    blas::swap(m, &A[i * lda], 1, &A[j * lda], 1);

    // swap idx array elements
    // Find idx element with value i and assign it to j
    auto it = std::find(idx.begin() + i, idx.begin() + k, i + 1);
    idx[it - (idx.begin())] = j + 1;
  }
}

} // namespace miniapps::util
#endif
