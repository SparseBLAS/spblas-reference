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

  auto&& [a_values, a_rowptr, a_colind, a_shape, as] = generate_csr<float>(m, k, nnz);
  auto&& [b_values, b_rowptr, b_colind, b_shape, bs] = generate_csr<float>(k, n, nnz);

  csr_view<float> a(a_values, a_rowptr, a_colind, a_shape, nnz);
  csr_view<float> b(b_values, b_rowptr, b_colind, b_shape, nnz);

  std::vector<float> c_values(m*n);
  std::vector<spblas::index_t> c_rowptr(m+1);
  std::vector<spblas::index_t> c_colind(m*n);

  csr_view<float> c(c_values, c_rowptr, c_colind, {m, n}, 0);

  multiply(a, b, c);

  for (auto&& [i, row] : __backend::rows(c)) {
    fmt::print("{}: {}\n", i, row);
  }

  return 0;
}
