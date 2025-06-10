#include <iostream>

#include <spblas/spblas.hpp>
#include <thrust/device_vector.h>

#include <fmt/core.h>
#include <fmt/printf.h>
#include <fmt/ranges.h>

int main(int argc, char** argv) {
  using namespace spblas;
  namespace md = spblas::__mdspan;

  using T = float;
  using index_t = spblas::index_t;
  using offset_t = spblas::offset_t;

  spblas::index_t m = 100;
  spblas::index_t n = 10;
  spblas::index_t k = 100;
  spblas::index_t nnz = 100;

  fmt::print("\n\t###########################################################"
             "######################");
  fmt::print("\n\t### Running SpGEMM Example:");
  fmt::print("\n\t###");
  fmt::print("\n\t###   C = A * B");
  fmt::print("\n\t###");
  fmt::print("\n\t### with ");
  fmt::print("\n\t### A, in CSR format, of size ({}, {}) with nnz = {}", m, k,
             nnz);
  fmt::print("\n\t### B, in CSR format, of size ({}, {}) with nnz = {}", k, n,
             nnz);
  fmt::print("\n\t### C, in CSR format, of size ({}, {}) with nnz to be"
             " determined",
             m, n);
  fmt::print("\n\t### using float and spblas::index_t (size = {} bytes)",
             sizeof(spblas::index_t));
  fmt::print("\n\t###########################################################"
             "######################");
  fmt::print("\n");

  // generate csr on CPU
  auto&& [a_values, a_rowptr, a_colind, a_shape, a_nnz] =
      generate_csr<T>(m, k, nnz);
  auto&& [b_values, b_rowptr, b_colind, b_shape, b_nnz] =
      generate_csr<T>(k, n, nnz);

  // copy the data to gpu
  thrust::device_vector<value_t> d_a_values(a_values);
  thrust::device_vector<offset_t> d_a_rowptr(a_rowptr);
  thrust::device_vector<index_t> d_a_colind(a_colind);
  // create Csr view A on device
  spblas::csr_view<value_t, index_t, offset_t> d_a(
      d_a_values.data().get(), d_a_rowptr.data().get(), d_a_colind.data().get(),
      a_shape, a_nnz);
  // copy the data to gpu
  thrust::device_vector<value_t> d_b_values(b_values);
  thrust::device_vector<offset_t> d_b_rowptr(b_rowptr);
  thrust::device_vector<index_t> d_b_colind(b_colind);
  spblas::csr_view<value_t, index_t, offset_t> d_b(
      d_b_values.data().get(), d_b_rowptr.data().get(), d_b_colind.data().get(),
      b_shape, b_nnz);
  // create Csr view B on device
  spblas::csr_view<value_t, index_t, offset_t> b(b_values, b_rowptr, b_colind,
                                                 b_shape, b_nnz);

  std::vector<offset_t> c_rowptr(m + 1);
  thrust::device_vector<offset_t> d_c_rowptr(c_rowptr);

  // create Csr view C
  spblas::csr_view<value_t, index_t, offset_t> d_c(
      nullptr, d_c_rowptr.data().get(), nullptr, {m, n}, 0);

  spblas::spgemm_state_t state;
  // symbolic compute -> give nnz
  spblas::multiply_symbolic_compute(state, d_a, d_b, d_c);
  auto nnz = state.result_nnz();
  // update Csr view with the allocation
  std::vector<value_t> c_values(nnz);
  std::vector<index_t> c_colind(nnz);
  thrust::device_vector<value_t> d_c_values(c_values);
  thrust::device_vector<index_t> d_c_colind(c_colind);
  std::span<value_t> d_c_values_span(d_c_values.data().get(), nnz);
  std::span<offset_t> d_c_rowptr_span(d_c_rowptr.data().get(), m + 1);
  std::span<index_t> d_c_colind_span(d_c_colind.data().get(), nnz);
  d_c.update(d_c_values_span, d_c_rowptr_span, d_c_colind_span, {m, n}, nnz);

  fmt::print("\t\t C_nnz = {}", nnz);

  // fill the sparsity of C
  spblas::multiply_symbolic_fill(state, d_a, d_b, d_c);

  for (int i = 0; i < 3; i++) {
    // we can redo it several time if A and B structure is not changed
    // update A and B value
    spblas::multiply_numeric(state, d_a, d_b, d_c);
  }

  fmt::print("\tExample is completed!\n");

  return 0;
}
