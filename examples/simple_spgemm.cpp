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
  spblas::index_t nnz = 100;

  std::cout << "\n\t###########################################################"
               "######################"
            << "\n\t### Running SpGEMM Example:"
            << "\n\t###"
            << "\n\t###   C = A * B"
            << "\n\t###"
            << "\n\t### with "
            << "\n\t### A in CSR format of size (" << m << ", " << k
            << ") with nnz = " << nnz << "\n\t### B in CSR format of size ("
            << k << ", " << n << ") with nnz = " << nnz
            << "\n\t### C in CSR format of size (" << m << ", " << n
            << ") with nnz to be determined"
            << "\n\t### using float and spblas::index_t (size = "
            << sizeof(spblas::index_t) << " bytes)"
            << "\n\t###########################################################"
               "######################"
            << std::endl;

  auto&& [a_values, a_rowptr, a_colind, a_shape, as] =
      generate_csr<float>(m, k, nnz);
  auto&& [b_values, b_rowptr, b_colind, b_shape, bs] =
      generate_csr<float>(k, n, nnz);

  csr_view<float> a(a_values, a_rowptr, a_colind, a_shape, nnz);
  csr_view<float> b(b_values, b_rowptr, b_colind, b_shape, nnz);

  std::vector<spblas::index_t> c_rowptr(m + 1);

  csr_view<float> c(nullptr, c_rowptr.data(), nullptr, {m, n}, 0);

  auto info = multiply_inspect(scaled(1.f, a), b, c);

  std::cout << "\t\t C_nnz = " << info.result_nnz() << std::endl;
  std::vector<float> c_values(info.result_nnz());
  std::vector<spblas::index_t> c_colind(info.result_nnz());
  c.update(c_values, c_rowptr, c_colind);

  multiply_execute(info, scaled(1.f, a), b, c);

  for (auto&& [i, row] : spblas::__backend::rows(c)) {
    fmt::print("{}: {}\n", i, row);
  }

  std::cout << "\tExample is completed!" << std::endl;

  return 0;
}
