#include <armpl_sparse.h>
#undef I
#include <spblas/spblas.hpp>

#include <fmt/ranges.h>

int main(int argc, char** argv) {
  using namespace spblas;

  using T = float;
  using I = std::int32_t;

  auto&& [values, rowptr, colind, shape, nnz] =
      generate_csr<T, I>(100, 100, 10);

  csr_view<T, I> a(values, rowptr, colind, shape, nnz);

  std::vector<T> b(100, 1);
  std::vector<T> c(100, 0);

  // multiply(a, b, c);

  for (auto&& [i, row] : __backend::rows(a)) {
    for (auto&& [k, v] : row) {
      c[i] += v * b[k];
    }
  }

  std::vector<T> c_armpl(100, 0);

  armpl_spmat_t a_handle;
  armpl_spmat_create_csr_s(&a_handle, a.shape()[0], a.shape()[1],
                           a.rowptr().data(), a.colind().data(),
                           a.values().data(), ARMPL_SPARSE_CREATE_NOCOPY);

  auto stat = armpl_spmv_exec_s(ARMPL_SPARSE_OPERATION_NOTRANS, 1.0f, a_handle,
                                b.data(), 0, c_armpl.data());

  fmt::print("c (ref): {}\n", c);
  fmt::print("c (arm): {}\n", c_armpl);

  return 0;
}
