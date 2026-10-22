#include <iostream>

#include <spblas/spblas.hpp>

#include <fmt/core.h>
#include <fmt/printf.h>
#include <fmt/ranges.h>

int main(int argc, char** argv) {
  using namespace spblas;
  namespace md = spblas::__mdspan;

  using T = float;

  spblas::index_t m = 100;
  spblas::index_t n = 10;
  spblas::index_t k = 100;
  spblas::index_t nnz = 100;

  fmt::print("\n\t###########################################################"
             "######################");
  fmt::print("\n\t### Running SpGEMM Example:");
  fmt::print("\n\t###");
  fmt::print("\n\t###   C = A * B");
  fmt::print("\n\t###");
  fmt::print("\n\t### with ");
  fmt::print("\n\t### A, in CSR format, of size ({}, {}) with nnz = {}", m, k,
             nnz);
  fmt::print("\n\t### B, in CSR format, of size ({}, {}) with nnz = {}", k, n,
             nnz);
  fmt::print("\n\t### C, in CSR format, of size ({}, {}) with nnz to be"
             " determined",
             m, n);
  fmt::print("\n\t### using float and spblas::index_t (size = {} bytes)",
             sizeof(spblas::index_t));
  fmt::print("\n\t###########################################################"
             "######################");
  fmt::print("\n");

  auto&& [a_values, a_rowptr, a_colind, a_shape, as] =
      generate_csr<T>(m, k, nnz);
  auto&& [b_values, b_rowptr, b_colind, b_shape, bs] =
      generate_csr<T>(k, n, nnz);

  csr_wrapper<T> a(a_values, a_rowptr, a_colind, a_shape, nnz);
  csr_wrapper<T> b(b_values, b_rowptr, b_colind, b_shape, nnz);
  // Inputs are wrapped in a matrix_opt, where the library may keep
  // optimization data for them across calls.
  matrix_opt a_opt(a);
  matrix_opt b_opt(b);

  // Output with unknown sparsity: only the shape is known up front.
  csr_wrapper<T> c(m, n);

  operation_state state;
  multiply_compute(state, 1.f, a_opt, b_opt, c);

  fmt::print("\t\t C_nnz = {}", state.get_result_nnz());

  // The user allocates C's arrays.
  std::vector<T> c_values(state.get_result_nnz());
  std::vector<spblas::index_t> c_rowptr(m + 1);
  std::vector<spblas::index_t> c_colind(state.get_result_nnz());
  c.update(c_values, c_rowptr, c_colind);

  multiply_fill(state, 1.f, a_opt, b_opt, c);

  for (auto&& [i, row] : spblas::__backend::rows(c)) {
    fmt::print("{}: {}\n", i, row);
  }

  fmt::print("\tExample is completed!\n");

  return 0;
}
