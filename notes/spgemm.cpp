#include <sparse_blas/sparse_blas.hpp>

int main(int argc, char** argv) {
  using namespace spblas;

  csr_matrix<float> a(/* ... */);
  csr_matrix<float> b(/* ... */);
  csr_matrix<float> c;

  auto info = multiply_inspect(a, b, c);

  // Allocate more memory for c based on `info`

  auto [values, rowptr, colind] = allocate_memory_for(info);

  // `info` also has implementation-specific optimization data.

  multiply_execute(info, a, b, c);

  // update_info_for_new_values(info, {a, left_operand_t});

  return 0;
}
