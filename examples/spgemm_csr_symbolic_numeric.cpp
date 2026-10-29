#include <algorithm>
#include <cmath>

#include <spblas/spblas.hpp>

#include <fmt/core.h>

// C = alpha * A * B, recomputed for several sets of A's values that share one
// sparsity pattern (e.g. a matrix assembled anew in every time step): the
// pattern of C is computed once, and each step only reruns the numeric stage.
int main(int argc, char** argv) {
  using namespace spblas;

  using T = double;
  using I = spblas::index_t;

  I m = 200;
  I k = 150;
  I n = 180;
  I nnz_in = 1500;
  int steps = 4;

  fmt::print("\n\t###########################################################"
             "######################");
  fmt::print("\n\t### Running SpGEMM Symbolic / Numeric Example:");
  fmt::print("\n\t###");
  fmt::print("\n\t###   C = alpha * A_s * B,  s = 0, ..., {}", steps - 1);
  fmt::print("\n\t###");
  fmt::print("\n\t### with ");
  fmt::print("\n\t### A_s, in CSR format, of size ({}, {}) with nnz = {};", m,
             k, nnz_in);
  fmt::print("\n\t###      new values in every step, same sparsity pattern");
  fmt::print("\n\t### B, in CSR format, of size ({}, {}) with nnz = {}", k, n,
             nnz_in);
  fmt::print("\n\t### C, in CSR format, of size ({}, {}): pattern computed "
             "once, values per step",
             m, n);
  fmt::print("\n\t### using double and spblas::index_t (size = {} bytes)",
             sizeof(I));
  fmt::print("\n\t###########################################################"
             "######################");
  fmt::print("\n");

  auto&& [a_values, a_rowptr, a_colind, a_shape, a_nnz] =
      generate_csr<T, I>(m, k, nnz_in, 0);
  auto&& [b_values, b_rowptr, b_colind, b_shape, b_nnz] =
      generate_csr<T, I>(k, n, nnz_in, 1);
  const std::vector<T> a_initial = a_values;

  // Inputs are wrapped in a matrix_opt, where the library may keep
  // optimization data for them across calls. A's values are changed in its
  // arrays below; the library is told with update_matrix(a), since its data
  // for A is derived from them.
  matrix_opt a(csr_wrapper<T, I>(a_values, a_rowptr, a_colind, a_shape, a_nnz));
  matrix_opt b(csr_wrapper<T, I>(b_values, b_rowptr, b_colind, b_shape, b_nnz));

  T alpha = 2;
  csr_wrapper<T, I> c(m, n); // only C's shape is known up front
  operation_state state;     // carries data between the stages below

  multiply_inspect(state, alpha, a, b, c);

  // 1. Compute how many entries C has.
  multiply_compute_symbolic(state, alpha, a, b, c);
  fmt::print("\n\t\t C_nnz = {}\n", state.get_result_nnz());

  // 2. Allocate C's pattern arrays and fill in the pattern. The values array
  //    may stay unallocated for now.
  std::vector<I> c_rowptr(m + 1);
  std::vector<I> c_colind(state.get_result_nnz());
  std::vector<T> c_values;
  c.update(c_values, c_rowptr, c_colind);
  multiply_fill_symbolic(state, alpha, a, b, c);

  // 3. Allocate C's values; every step now only runs the numeric stage.
  c_values.resize(state.get_result_nnz());
  c.update(c_values, c_rowptr, c_colind);

  // Each step is checked without relying on SpGEMM: C x = alpha A (B x).
  std::vector<T> x(n);
  for (I j = 0; j < n; j++) {
    x[j] = T(j % 7 + 1) / 7;
  }
  std::vector<T> bx(k), abx(m), cx(m);
  operation_state check_state;

  for (int s = 0; s < steps; s++) {
    // New values for A, same pattern.
    for (std::size_t p = 0; p < a_values.size(); p++) {
      a_values[p] = a_initial[p] * (1 + T(0.25) * s * T(p % 3));
    }
    update_matrix(a);

    multiply_numeric(state, alpha, a, b, c);

    multiply(check_state, T(1), matrix_opt(c), x, cx);
    multiply(check_state, T(1), b, x, bx);
    multiply(check_state, alpha, a, bx, abx);

    T error = 0;
    T scale = 0;
    for (I i = 0; i < m; i++) {
      error = std::max(error, std::abs(cx[i] - abx[i]));
      scale = std::max(scale, std::abs(abx[i]));
    }
    fmt::print("\t\t step {}: max |C x - alpha A (B x)| / max |alpha A (B x)| "
               "= {:.2e}\n",
               s, scale > 0 ? error / scale : error);
  }

  fmt::print("\tExample is completed!\n");

  return 0;
}
