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

  auto&& [values, rowptr, colind, shape, _] = generate_csr<float>(m, k, nnz);

  csr_view<float> a(values, rowptr, colind, shape, nnz);

  std::vector<float> b_values(k * n, 1);
  std::vector<float> c_values(m * n, 0);

  md::mdspan b(b_values.data(), k, n);
  md::mdspan c(c_values.data(), m, n);

  auto s = spblas::__backend::shape(c);

  scale(45, b);

  fmt::print("{}, {}\n", s[0], s[1]);

  fmt::print("{}\n", spblas::__backend::values(b));

  std::vector<float> v(n, 1);
  std::vector<float> v_out(k, 0);

  multiply(b, v, v_out);

  return 0;
}
