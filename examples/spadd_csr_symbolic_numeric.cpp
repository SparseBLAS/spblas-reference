#include <algorithm>
#include <cmath>

#include <spblas/spblas.hpp>

#include <fmt/core.h>

// C = A + B_s for several sets of B's values that share one sparsity pattern:
// the pattern of C (the union of A's and B's) is computed once, and each step
// only reruns the numeric stage.
int main(int argc, char** argv) {
  using namespace spblas;

  using T = double;
  using I = spblas::index_t;

  I m = 300;
  I n = 250;
  I nnz_in = 3000;
  int steps = 4;

  fmt::print("\n\t###########################################################"
             "######################");
  fmt::print("\n\t### Running SpADD Symbolic / Numeric Example:");
  fmt::print("\n\t###");
  fmt::print("\n\t###   C = A + B_s,  s = 0, ..., {}", steps - 1);
  fmt::print("\n\t###");
  fmt::print("\n\t### with ");
  fmt::print("\n\t### A, in CSR format, of size ({}, {}) with nnz = {}", m, n,
             nnz_in);
  fmt::print("\n\t### B_s, in CSR format, of size ({}, {}) with nnz = {};", m,
             n, nnz_in);
  fmt::print("\n\t###      new values in every step, same sparsity pattern");
  fmt::print("\n\t### C, in CSR format, of size ({}, {}): pattern computed "
             "once, values per step",
             m, n);
  fmt::print("\n\t### using double and spblas::index_t (size = {} bytes)",
             sizeof(I));
  fmt::print("\n\t###########################################################"
             "######################");
  fmt::print("\n");

  auto&& [a_values, a_rowptr, a_colind, a_shape, a_nnz] =
      generate_csr<T, I>(m, n, nnz_in, 0);
  auto&& [b_values, b_rowptr, b_colind, b_shape, b_nnz] =
      generate_csr<T, I>(m, n, nnz_in, 1);
  const std::vector<T> b_initial = b_values;

  // Inputs are wrapped in a matrix_opt, where the library may keep
  // optimization data for them across calls. B's values are changed in its
  // arrays below; the library is told with update_matrix(b), since its data
  // for B is derived from them.
  matrix_opt a(csr_wrapper<T, I>(a_values, a_rowptr, a_colind, a_shape, a_nnz));
  matrix_opt b(csr_wrapper<T, I>(b_values, b_rowptr, b_colind, b_shape, b_nnz));

  csr_wrapper<T, I> c(m, n); // only C's shape is known up front
  operation_state state;     // carries data between the stages below

  add_inspect(state, a, b, c);

  // 1. Compute how many entries C has.
  add_compute_symbolic(state, a, b, c);
  fmt::print("\n\t\t C_nnz = {}\n", state.get_result_nnz());

  // 2. Allocate C's pattern arrays and fill in the pattern. The values array
  //    may stay unallocated for now.
  std::vector<I> c_rowptr(m + 1);
  std::vector<I> c_colind(state.get_result_nnz());
  std::vector<T> c_values;
  c.update(c_values, c_rowptr, c_colind);
  add_fill_symbolic(state, a, b, c);

  // 3. Allocate C's values; every step now only runs the numeric stage.
  c_values.resize(state.get_result_nnz());
  c.update(c_values, c_rowptr, c_colind);

  // Each step is checked without relying on SpADD: C x = A x + B x.
  std::vector<T> x(n);
  for (I j = 0; j < n; j++) {
    x[j] = T(j % 5 + 1) / 5;
  }
  std::vector<T> ax(m), abx(m), cx(m);
  operation_state check_state;

  for (int s = 0; s < steps; s++) {
    // New values for B, same pattern.
    for (std::size_t p = 0; p < b_values.size(); p++) {
      b_values[p] = b_initial[p] * (1 - T(0.3) * s * T(p % 2));
    }
    update_matrix(b);

    add_numeric(state, a, b, c);

    multiply(check_state, T(1), matrix_opt(c), x, cx);
    multiply(check_state, T(1), a, x, ax);
    multiply(check_state, T(1), b, x, T(1), ax, abx); // abx = B x + A x

    T error = 0;
    T scale = 0;
    for (I i = 0; i < m; i++) {
      error = std::max(error, std::abs(cx[i] - abx[i]));
      scale = std::max(scale, std::abs(abx[i]));
    }
    fmt::print("\t\t step {}: max |C x - (A x + B x)| / max |A x + B x| = "
               "{:.2e}\n",
               s, scale > 0 ? error / scale : error);
  }

  fmt::print("\tExample is completed!\n");

  return 0;
}
