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

  auto&& [a_values, a_rowptr, a_colind, a_shape, as] =
      generate_csr<float>(m, k, nnz);
  auto&& [b_values, b_rowptr, b_colind, b_shape, bs] =
      generate_csr<float>(k, n, nnz);

  csr_view<float> a(a_values, a_rowptr, a_colind, a_shape, nnz);
  csr_view<float> b(b_values, b_rowptr, b_colind, b_shape, nnz);

  csr_view<float> c(nullptr, nullptr, nullptr, {m, n}, 0);

  auto info = multiply_inspect(a, b, c);

  std::vector<float> c_values(info.result_nnz());
  std::vector<spblas::index_t> c_rowptr(info.result_shape()[0] + 1);
  std::vector<spblas::index_t> c_colind(info.result_nnz());

  c.update(c_values, c_rowptr, c_colind);

  multiply_execute(info, a, b, c);

  for (auto&& [i, row] : spblas::__backend::rows(c)) {
    fmt::print("{}: {}\n", i, row);
  }

  return 0;
}
