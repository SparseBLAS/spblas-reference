#include <spblas/spblas.hpp>

#include <fmt/core.h>
#include <fmt/ranges.h>

int main(int argc, char** argv) {
  using namespace spblas;

  using T = float;

  spblas::index_t m = 10;
  spblas::index_t n = 10;
  spblas::index_t nnz = 20;

  fmt::print("\n\t###########################################################"
             "######################");
  fmt::print("\n\t### Running SpADD Example:");
  fmt::print("\n\t###");
  fmt::print("\n\t###   C = A + B");
  fmt::print("\n\t###");
  fmt::print("\n\t### with ");
  fmt::print("\n\t### A, in CSR format, of size ({}, {}) with nnz = {}", m, n,
             nnz);
  fmt::print("\n\t### B, in CSR format, of size ({}, {}) with nnz = {}", m, n,
             nnz);
  fmt::print("\n\t### C, in CSR format, of size ({}, {}) with nnz to be"
             " determined",
             m, n);
  fmt::print("\n\t### using float and spblas::index_t (size = {} bytes)",
             sizeof(spblas::index_t));
  fmt::print("\n\t###########################################################"
             "######################");
  fmt::print("\n");

  auto&& [a_values, a_rowptr, a_colind, a_shape, a_nnz] =
      generate_csr<T>(m, n, nnz, 0);
  auto&& [b_values, b_rowptr, b_colind, b_shape, b_nnz] =
      generate_csr<T>(m, n, nnz, 1);

  csr_wrapper<T> a(a_values, a_rowptr, a_colind, a_shape, a_nnz);
  csr_wrapper<T> b(b_values, b_rowptr, b_colind, b_shape, b_nnz);
  // Inputs are wrapped in a matrix_opt, where the library may keep
  // optimization data for them across calls.
  matrix_opt a_opt(a);
  matrix_opt b_opt(b);

  // Output with unknown sparsity: only the shape is known up front.
  csr_wrapper<T> c(m, n);

  operation_state state;
  add_compute(state, a_opt, b_opt, c);

  fmt::print("\t\t C_nnz = {}\n", state.get_result_nnz());

  // The user allocates C's arrays.
  std::vector<T> c_values(state.get_result_nnz());
  std::vector<spblas::index_t> c_rowptr(m + 1);
  std::vector<spblas::index_t> c_colind(state.get_result_nnz());
  c.update(c_values, c_rowptr, c_colind);

  add_fill(state, a_opt, b_opt, c);

  // The order within C's rows is backend-defined; sort them for printing.
  // C.sort_state() says what the backend knows; if it is already `sorted`,
  // sort_matrix does nothing.
  fmt::print("\tC is {}sorted\n",
             c.sort_state() == sort_state::sorted ? "" : "not known to be ");
  operation_state sorting;
  sort_matrix(sorting, c);

  for (auto&& [i, row] : spblas::__backend::rows(c)) {
    fmt::print("{}: {}\n", i, row);
  }

  fmt::print("\tExample is completed!\n");

  return 0;
}
