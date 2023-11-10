#include <spblas/spblas.hpp>

int main(int argc, char** argv) {
  using namespace spblas;

  using T = float;
  using I = index_t;
  using O = index_t;

  T* values = (T *) nullptr;
  O* rowptr = (O *) nullptr;
  I* colind = (I *) nullptr;

  csr_view<float> v(values, rowptr, colind, {100, 100}, 10);

  return 0;
}