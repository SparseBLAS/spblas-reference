#include <spblas/spblas.hpp>

int main(int argc, char** argv) {
  using namespace spblas;

  csr_view<float> v(100, 100, 10, nullptr, nullptr, nullptr);

  return 0;
}