#include <spblas/spblas.hpp>

int main(int argc, char** argv) {
  using namespace spblas;

  auto&& [values, rowptr, colind, shape, nnz] =
      generate_csr<float>(100, 100, 10);

  csr_view<float> v(values, rowptr, colind, shape, nnz);

  scale(1000.f, v);

  std::vector<float> b(100, 1);
  std::vector<float> c(100, 0);

  multiply(v, b, c);

  return 0;
}
