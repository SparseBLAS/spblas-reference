#include <iostream>
#include <spblas/spblas.hpp>

#include <fmt/core.h>
#include <fmt/ranges.h>

int main(int argc, char** argv) {
  using namespace spblas;

  using DATA_TYPE = float;

  spblas::index_t m = 100;
  spblas::index_t n = 100;
  spblas::index_t nnz_in = 10;

  std::cout << "\n\t###########################################################"
               "######################"
            << "\n\t### Running SpMV Example:"
            << "\n\t###"
            << "\n\t###   y = alpha * A * x"
            << "\n\t###"
            << "\n\t### with "
            << "\n\t### A in CSR format of size (" << m << ", " << n
            << ") with nnz = " << nnz_in
            << "\n\t### x, a dense vector of size (" << n << ", " << 1 << ")"
            << "\n\t### y, a dense vector of size (" << m << ", " << 1 << ")"
            << "\n\t### using float and spblas::index_t (size = "
            << sizeof(spblas::index_t) << " bytes)"
            << "\n\t###########################################################"
               "######################"
            << std::endl;

  auto&& [values, rowptr, colind, shape, nnz] =
      generate_csr<DATA_TYPE, spblas::index_t>(m, n, nnz_in);

  csr_view<DATA_TYPE, spblas::index_t> a(values, rowptr, colind, shape, nnz);

  // Scale every value of `a` by 5 in place.
  // scale(5.f, a);

  std::vector<DATA_TYPE> x(n, 1);
  std::vector<DATA_TYPE> y(m, 0);

  DATA_TYPE alpha = 1.2f;
  auto a_scaled = scaled(alpha, a);

  // y = alpha * A * x
  multiply(a_scaled, x, y);

  std::cout << "\tExample is completed!" << std::endl;

  return 0;
}
