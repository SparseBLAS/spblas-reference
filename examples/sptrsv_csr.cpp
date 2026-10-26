#include <spblas/spblas.hpp>

#include <fmt/core.h>
#include <fmt/ranges.h>

int main(int argc, char** argv) {
  using namespace spblas;

  using T = float;

  spblas::index_t m = 100;
  spblas::index_t nnz_in = 20;

  fmt::print("\n\t###########################################################"
             "######################");
  fmt::print("\n\t### Running SpTRSV Example:");
  fmt::print("\n\t###");
  fmt::print("\n\t###   solve for x:  A * x = alpha * b");
  fmt::print("\n\t###");
  fmt::print("\n\t### with ");
  fmt::print("\n\t### A, in CSR format, of size ({}, {}) with nnz = {}", m, m,
             nnz_in);
  fmt::print("\n\t### x, a dense vector, of size ({}, {})", m, 1);
  fmt::print("\n\t### b, a dense vector, of size ({}, {})", m, 1);
  fmt::print("\n\t### using float and spblas::index_t (size = {} bytes)",
             sizeof(spblas::index_t));
  fmt::print("\n\t###########################################################"
             "######################");
  fmt::print("\n");

  auto&& [values, rowptr, colind, shape, nnz] =
      generate_csr_sorted<T, spblas::index_t>(m, m, nnz_in);

  // scale values of matrix to make the implicit unit diagonal matrix
  // be diagonally dominant, so it is solveable
  T scale_factor = 1e-3f;
  std::transform(values.begin(), values.end(), values.begin(),
                 [scale_factor](T val) { return scale_factor * val; });

  // Properties of the data the wrapper holds: its rows are sorted.
  property_list props(sort_state::sorted);
  csr_wrapper<T, spblas::index_t> a(values, rowptr, colind, shape, nnz, props);
  // Inputs are wrapped in a matrix_opt, where the library may keep
  // optimization data for them across calls.
  matrix_opt a_opt(a);

  operation_state state;

  // Scale every value of `a` by 5 in place.
  // scale(state, 5.f, a_opt);

  std::vector<T> x(m, 0);
  std::vector<T> b(m, 1);

  T alpha = 1.2f;
  // solve for x:  lower(A) * x = alpha * b
  triangular_solve(
      state,
      view::triangle(a_opt, view::uplo::lower, view::diag::implicit_unit),
      alpha, b, x);

  fmt::print("\tExample is completed!\n");

  return 0;
}
