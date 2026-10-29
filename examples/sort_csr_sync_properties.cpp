#include <algorithm>
#include <stdexcept>
#include <vector>

#include <spblas/spblas.hpp>

#include <fmt/core.h>
#include <fmt/ranges.h>

// Whether every row's column indices are in increasing order.
template <typename W>
bool rows_are_sorted(W&& w) {
  for (auto&& [i, row] : spblas::__backend::rows(w)) {
    std::vector<spblas::tensor_index_t<W>> cols;
    for (auto&& [j, v] : row) {
      cols.push_back(j);
    }
    if (!std::is_sorted(cols.begin(), cols.end())) {
      return false;
    }
  }
  return true;
}

const char* name(spblas::sort_state s) {
  switch (s) {
  case spblas::sort_state::sorted:
    return "sorted";
  case spblas::sort_state::unsorted:
    return "unsorted";
  default:
    return "unknown";
  }
}

int main(int argc, char** argv) {
  using namespace spblas;

  using T = float;

  spblas::index_t m = 100;
  spblas::index_t n = 100;
  spblas::index_t nnz_in = 1000;

  fmt::print("\n\t###########################################################"
             "######################");
  fmt::print("\n\t### Running sort_matrix / sync_properties Example:");
  fmt::print("\n\t###");
  fmt::print("\n\t###   sort A's rows through its matrix_opt, then copy the");
  fmt::print("\n\t###   properties the matrix_opt learned back into A");
  fmt::print("\n\t###");
  fmt::print("\n\t### with ");
  fmt::print("\n\t### A, in CSR format, of size ({}, {}) with nnz = {}", m, n,
             nnz_in);
  fmt::print("\n\t### using float and spblas::index_t (size = {} bytes)",
             sizeof(spblas::index_t));
  fmt::print("\n\t###########################################################"
             "######################");
  fmt::print("\n");

  // Rows in random order.
  auto&& [values, rowptr, colind, shape, nnz] =
      generate_csr<T, spblas::index_t>(m, n, nnz_in);

  csr_wrapper<T, spblas::index_t> a(values, rowptr, colind, shape, nnz);
  // A matrix_opt keeps its own copy of the wrapper (always by value), shared
  // by all copies of the matrix_opt; properties the library learns about the
  // data are recorded there.
  matrix_opt a_opt(a);

  fmt::print("\n\tbefore sorting:");
  fmt::print("\n\t\ta_opt.sort_state() = {}", name(a_opt.sort_state()));
  fmt::print("\n\t\ta.sort_state()     = {}", name(a.sort_state()));
  fmt::print("\n\t\trows sorted        = {}", rows_are_sorted(a));

  operation_state state;
  std::vector<T> x(n, 1);
  std::vector<T> y(m, 0);
  multiply(state, 1.f, a_opt, x, y);

  // Sort the user's arrays in place through the matrix_opt. Its copy of the
  // wrapper is now `sorted`; the user's wrapper `a` is a separate copy and
  // still says `unknown`, although the arrays it wraps are now sorted.
  operation_state sorting;
  sort_matrix(sorting, a_opt);
  fmt::print("\n\tafter sort_matrix(state, a_opt):");
  fmt::print("\n\t\ta_opt.sort_state() = {}", name(a_opt.sort_state()));
  fmt::print("\n\t\ta.sort_state()     = {}", name(a.sort_state()));
  fmt::print("\n\t\trows sorted        = {}", rows_are_sorted(a));

  // Copy the properties back. This checks that a_opt is over a's arrays
  // (same spans, shape and nnz) and changes only a's properties.
  a.sync_properties(a_opt);
  fmt::print("\n\tafter a.sync_properties(a_opt):");
  fmt::print("\n\t\ta.sort_state()     = {}", name(a.sort_state()));

  // The synced wrapper carries the property into whatever is built from it
  // later, e.g. another matrix_opt, where sort_matrix now has nothing to do.
  matrix_opt a_opt2(a);
  sort_matrix(sorting, a_opt2);
  fmt::print("\n\ta new matrix_opt from a: a_opt2.sort_state() = {}",
             name(a_opt2.sort_state()));

  // A matrix_opt over other arrays (here, a copy of A's) has properties that
  // are claims about other data, so syncing from it throws and leaves a's
  // properties as they are.
  std::vector<spblas::index_t> colind_copy(colind);
  csr_wrapper<T, spblas::index_t> b(values, rowptr, colind_copy, shape, nnz);
  matrix_opt b_opt(b);
  try {
    a.sync_properties(b_opt);
  } catch (const std::invalid_argument& e) {
    fmt::print("\n\ta.sync_properties(b_opt) throws: {}", e.what());
  }
  fmt::print("\n\t\ta.sort_state()     = {}\n", name(a.sort_state()));

  fmt::print("\tExample is completed!\n");

  return 0;
}
