#include <spblas/spblas.hpp>

#include <fmt/core.h>
#include <fmt/ranges.h>

int main(int argc, char** argv) {
  using namespace spblas;
  namespace md = spblas::__mdspan;

  using T = float;

  spblas::index_t m = 10;
  spblas::index_t n = 10;
  spblas::index_t k = 10;
  spblas::index_t nnz_in = 20;

  fmt::print("\n\t###########################################################"
             "######################");
  fmt::print("\n\t### Running Advanced SpMM Example:");
  fmt::print("\n\t###");
  fmt::print("\n\t###   Y = alpha * A * X");
  fmt::print("\n\t###");
  fmt::print("\n\t### with ");
  fmt::print("\n\t### A, in CSR format, of size ({}, {}) with nnz = {}", m, k,
             nnz_in);
  fmt::print("\n\t### x, a dense matrix, of size ({}, {})", k, n);
  fmt::print("\n\t### y, a dense vector, of size ({}, {})", m, n);
  fmt::print("\n\t### using float and spblas::index_t (size = {} bytes)",
             sizeof(spblas::index_t));
  fmt::print("\n\t###########################################################"
             "######################");
  fmt::print("\n");

  auto&& [values, rowptr, colind, shape, nnz] = generate_csr<T>(m, k, nnz_in);

  csr_view<T> a(values, rowptr, colind, shape, nnz);
  matrix_opt a_opt(a);

  std::vector<T> x_values(k * n, 1);
  std::vector<T> y_values(m * n, 0);

  md::mdspan x(x_values.data(), k, n);
  md::mdspan y(y_values.data(), m, n);

  // Y = A * X
  auto state = multiply_inspect(a_opt, x, y);
  multiply(state, a_opt, x, y);

  fmt::print("{}\n", spblas::__backend::values(y));

  fmt::print("\tExample is completed!\n");

  return 0;
}
