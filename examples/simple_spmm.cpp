#include <iostream>
#include <spblas/spblas.hpp>

#include <fmt/core.h>
#include <fmt/ranges.h>

int main(int argc, char** argv) {
  using namespace spblas;
  namespace md = spblas::__mdspan;

  spblas::index_t m = 100;
  spblas::index_t n = 10;
  spblas::index_t k = 100;
  spblas::index_t nnz = 10;

  std::cout << "\n\t###########################################################"
               "######################"
            << "\n\t### Running SpMM Example:"
            << "\n\t###"
            << "\n\t###   Y = alpha * A * X"
            << "\n\t###"
            << "\n\t### with "
            << "\n\t### A in CSR format of size (" << m << ", " << k
            << ") with nnz = " << nnz
            << "\n\t### X, a dense matrix in rowmajor format of size (" << k
            << ", " << n << ")"
            << "\n\t### Y, a dense matrix in rowmajor format of size (" << m
            << ", " << n << ")"
            << "\n\t### using float and spblas::index_t (size = "
            << sizeof(spblas::index_t) << " bytes)"
            << "\n\t###########################################################"
               "######################"
            << std::endl;

  auto&& [values, rowptr, colind, shape, _] = generate_csr<float>(m, k, nnz);

  csr_view<float> a(values, rowptr, colind, shape, nnz);

  std::vector<float> x_values(k * n, 1);
  std::vector<float> y_values(m * n, 0);

  md::mdspan x(x_values.data(), k, n);
  md::mdspan y(y_values.data(), m, n);

  auto a_view = scaled(2.f, a);

  // y = A * (alpha * x)
  multiply(a_view, scaled(2.f, x), y);

  fmt::print("{}\n", spblas::__backend::values(y));

  std::cout << "\tExample is completed!" << std::endl;

  return 0;
}
