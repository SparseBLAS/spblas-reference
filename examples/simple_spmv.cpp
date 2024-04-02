#include <spblas/spblas.hpp>

#include <fmt/ranges.h>

int main(int argc, char** argv) {
  using namespace spblas;

  auto&& [values, rowptr, colind, shape, nnz] =
      generate_csr<float>(100, 100, 10);

  csr_view<float> a(values, rowptr, colind, shape, nnz);

  // Scale every value of `a` by 5 in place.
  // scale(5.f, a);

  std::vector<float> b(100, 1);
  std::vector<float> c(100, 0);

  float alpha = 1.2f;
  auto a_scaled = scaled(alpha, a);

  // c = a * alpha * b
  multiply(a_scaled, b, c);

  return 0;
}
