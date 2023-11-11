#include <sparse_blas/sparse_blas.hpp>

int main(int argc, char** argv) {
  using namespace spblas;

  csr_matrix<float> a(/* ... */);
  dense_vector<float> x(/* ... */);
  dense_vector<float> y;

  operation_info_t info;

  device_policy policy;

  multiply_inspect(info, policy, a, x, y);
  multiply_inspect(info, policy, transposed(a), x, y);

  // Allocate more memory for y based on `info`

  while (/* ... */) {
    multiply_execute(info, policy, a, x, y);
    // do something with y, update x...
    multiply_execute(info, policy, transposed(a), y, x);
    // Maybe do some more stuff...
  }

  return 0;
}
