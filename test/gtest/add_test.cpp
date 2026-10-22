#include <gtest/gtest.h>

#include "util.hpp"
#include <spblas/spblas.hpp>

TEST(Add, CsrCsrCsr) {
  using T = float;
  using I = spblas::index_t;

  for (auto&& [m, n, nnz] : util::dims) {
    auto [a_values, a_rowptr, a_colind, a_shape, a_nnz] =
        spblas::generate_csr<T, I>(m, n, nnz, 0);
    auto [b_values, b_rowptr, b_colind, b_shape, b_nnz] =
        spblas::generate_csr<T, I>(m, n, nnz, 1);

    spblas::csr_wrapper<T, I> a(a_values, a_rowptr, a_colind, a_shape, a_nnz);
    spblas::csr_wrapper<T, I> b(b_values, b_rowptr, b_colind, b_shape, b_nnz);
    spblas::csr_wrapper<T, I> c(m, n);

    spblas::operation_state state;
    spblas::add_inspect(state, a, b, c);
    spblas::add_compute(state, a, b, c);

    std::vector<T> c_values(state.get_result_nnz());
    std::vector<I> c_colind(state.get_result_nnz());
    std::vector<I> c_rowptr(m + 1);
    c.update(c_values, c_rowptr, c_colind);

    spblas::add_fill(state, a, b, c);
    util::sort_output(c);

    // Pattern is the union of both patterns; values are A + B.
    auto a_dense = util::to_dense(a);
    auto b_dense = util::to_dense(b);
    std::size_t union_positions = 0;
    for (std::size_t i = 0; i < a_dense.size(); i++) {
      union_positions += (a_dense[i] != T(0) || b_dense[i] != T(0)) ? 1 : 0;
    }
    EXPECT_EQ(state.get_result_nnz(), util::count_positions(c));
    EXPECT_EQ(c_rowptr.back(), state.get_result_nnz());

    auto c_dense = util::to_dense(c);
    for (std::size_t i = 0; i < c_dense.size(); i++) {
      EXPECT_EQ_(a_dense[i] + b_dense[i], c_dense[i]);
    }
    // Generated values are nonzero, so value positions equal the pattern.
    EXPECT_EQ(state.get_result_nnz(), union_positions);
  }
}

TEST(Add, MatrixOptAndPolicy) {
  using T = double;
  using I = spblas::index_t;

  auto [m, n, nnz] = util::dims[2];
  auto [a_values, a_rowptr, a_colind, a_shape, a_nnz] =
      spblas::generate_csr<T, I>(m, n, nnz, 0);
  auto [b_values, b_rowptr, b_colind, b_shape, b_nnz] =
      spblas::generate_csr<T, I>(m, n, nnz, 1);

  spblas::matrix_opt a(
      spblas::csr_wrapper<T, I>(a_values, a_rowptr, a_colind, a_shape, a_nnz));
  spblas::matrix_opt b(
      spblas::csr_wrapper<T, I>(b_values, b_rowptr, b_colind, b_shape, b_nnz));
  spblas::csr_wrapper<T, I> c(m, n);

  spblas::operation_state state;
  spblas::add_inspect(util::test_policy{}, state, a, b, c);
  spblas::add_compute(util::test_policy{}, state, a, b, c);

  std::vector<T> c_values(state.get_result_nnz());
  std::vector<I> c_colind(state.get_result_nnz());
  std::vector<I> c_rowptr(m + 1);
  c.update(c_values, c_rowptr, c_colind);

  spblas::add_fill(util::test_policy{}, state, a, b, c);
  util::sort_output(c);

  auto a_dense = util::to_dense(a.base());
  auto b_dense = util::to_dense(b.base());
  auto c_dense = util::to_dense(c);
  for (std::size_t i = 0; i < c_dense.size(); i++) {
    EXPECT_EQ_(a_dense[i] + b_dense[i], c_dense[i]);
  }
}

TEST(Add, ThreeStage) {
  using T = float;
  using I = spblas::index_t;

  for (auto&& [m, n, nnz] : util::dims) {
    auto [a_values, a_rowptr, a_colind, a_shape, a_nnz] =
        spblas::generate_csr<T, I>(m, n, nnz, 0);
    auto [b_values, b_rowptr, b_colind, b_shape, b_nnz] =
        spblas::generate_csr<T, I>(m, n, nnz, 1);

    spblas::csr_wrapper<T, I> a(a_values, a_rowptr, a_colind, a_shape, a_nnz);
    spblas::csr_wrapper<T, I> b(b_values, b_rowptr, b_colind, b_shape, b_nnz);
    spblas::csr_wrapper<T, I> c(m, n);

    spblas::operation_state state;
    spblas::add_compute_symbolic(state, a, b, c);

    // Pattern first; values attached afterwards.
    std::vector<I> c_rowptr(m + 1);
    std::vector<I> c_colind(state.get_result_nnz());
    std::vector<T> c_values;
    c.update(c_values, c_rowptr, c_colind);
    spblas::add_fill_symbolic(state, a, b, c);
    EXPECT_EQ(c_rowptr.back(), state.get_result_nnz());
    EXPECT_EQ(util::count_positions(c), state.get_result_nnz());

    c_values.assign(state.get_result_nnz(), T(-1));
    c.update(c_values, c_rowptr, c_colind);

    for (int round = 0; round < 2; round++) {
      // Second round: new values for A, same pattern.
      if (round == 1) {
        for (auto&& v : a_values) {
          v *= T(-2);
        }
      }
      spblas::add_numeric(state, a, b, c);
      util::sort_output(c);

      auto a_dense = util::to_dense(a);
      auto b_dense = util::to_dense(b);
      auto c_dense = util::to_dense(c);
      for (std::size_t i = 0; i < c_dense.size(); i++) {
        EXPECT_EQ_(a_dense[i] + b_dense[i], c_dense[i]);
      }
    }
  }
}

TEST(Add, ThreeStageMatrixOptAndPolicy) {
  using T = double;
  using I = spblas::index_t;

  auto [m, n, nnz] = util::dims[2];
  auto [a_values, a_rowptr, a_colind, a_shape, a_nnz] =
      spblas::generate_csr<T, I>(m, n, nnz, 0);
  auto [b_values, b_rowptr, b_colind, b_shape, b_nnz] =
      spblas::generate_csr<T, I>(m, n, nnz, 1);
  spblas::matrix_opt a(
      spblas::csr_wrapper<T, I>(a_values, a_rowptr, a_colind, a_shape, a_nnz));
  spblas::matrix_opt b(
      spblas::csr_wrapper<T, I>(b_values, b_rowptr, b_colind, b_shape, b_nnz));
  spblas::csr_wrapper<T, I> c(m, n);

  spblas::operation_state state;
  spblas::add_compute_symbolic(util::test_policy{}, state, a, b, c);
  std::vector<I> c_rowptr(m + 1), c_colind(state.get_result_nnz());
  std::vector<T> c_values(state.get_result_nnz());
  c.update(c_values, c_rowptr, c_colind);
  spblas::add_fill_symbolic(util::test_policy{}, state, a, b, c);
  spblas::add_numeric(util::test_policy{}, state, a, b, c);
  util::sort_output(c);

  auto a_dense = util::to_dense(a.base());
  auto b_dense = util::to_dense(b.base());
  auto c_dense = util::to_dense(c);
  for (std::size_t i = 0; i < c_dense.size(); i++) {
    EXPECT_EQ_(a_dense[i] + b_dense[i], c_dense[i]);
  }
}

TEST(Add, CscCscCsc) {
  using T = float;
  using I = spblas::index_t;

  for (auto&& [m, n, nnz] : util::dims) {
    auto [a_values, a_colptr, a_rowind, a_shape, a_nnz] =
        spblas::generate_csc<T, I>(m, n, nnz, 0);
    auto [b_values, b_colptr, b_rowind, b_shape, b_nnz] =
        spblas::generate_csc<T, I>(m, n, nnz, 1);

    spblas::csc_wrapper<T, I> a(a_values, a_colptr, a_rowind, a_shape, a_nnz);
    spblas::csc_wrapper<T, I> b(b_values, b_colptr, b_rowind, b_shape, b_nnz);
    auto a_dense = util::to_dense(a);
    auto b_dense = util::to_dense(b);

    // 2-stage, with matrix_opt inputs.
    {
      spblas::matrix_opt a_opt(a);
      spblas::csc_wrapper<T, I> c(m, n);
      spblas::operation_state state;
      spblas::add_inspect(state, a_opt, b, c);
      spblas::add_compute(state, a_opt, b, c);

      std::vector<T> c_values(state.get_result_nnz());
      std::vector<I> c_rowind(state.get_result_nnz());
      std::vector<I> c_colptr(n + 1);
      c.update(c_values, c_colptr, c_rowind);
      spblas::add_fill(state, a_opt, b, c);
      util::sort_output(c);

      EXPECT_EQ(c_colptr.back(), state.get_result_nnz());
      EXPECT_EQ(util::count_positions(c), state.get_result_nnz());
      auto c_dense = util::to_dense(c);
      for (std::size_t i = 0; i < c_dense.size(); i++) {
        EXPECT_EQ_(a_dense[i] + b_dense[i], c_dense[i]);
      }
    }

    // 3-stage, with a policy.
    {
      spblas::csc_wrapper<T, I> c(m, n);
      spblas::operation_state state;
      spblas::add_compute_symbolic(util::test_policy{}, state, a, b, c);

      std::vector<I> c_colptr(n + 1);
      std::vector<I> c_rowind(state.get_result_nnz());
      std::vector<T> c_values;
      c.update(c_values, c_colptr, c_rowind);
      spblas::add_fill_symbolic(util::test_policy{}, state, a, b, c);

      c_values.assign(state.get_result_nnz(), T(-1));
      c.update(c_values, c_colptr, c_rowind);
      spblas::add_numeric(util::test_policy{}, state, a, b, c);
      util::sort_output(c);

      auto c_dense = util::to_dense(c);
      for (std::size_t i = 0; i < c_dense.size(); i++) {
        EXPECT_EQ_(a_dense[i] + b_dense[i], c_dense[i]);
      }
    }
  }
}

namespace {

template <typename A, typename B, typename C>
constexpr bool can_add = requires(spblas::operation_state& s, A a, B b, C c) {
  spblas::add_compute(s, a, b, c);
  spblas::add_fill(s, a, b, c);
  spblas::add_numeric(s, a, b, c);
};

} // namespace

// A^T + B with A in CSR (so A^T is CSC) and B, C in CSR. Not every backend
// supports mixed orientations (the reference kernel does not). A template, so
// that the unsupported branch is never instantiated.
template <typename T, typename I>
void transposed_operand_add_test() {
  auto [m, n, nnz] = util::dims[1];
  auto [a_values, a_rowptr, a_colind, a_shape, a_nnz] =
      spblas::generate_csr<T, I>(n, m, nnz, 0);
  auto [b_values, b_rowptr, b_colind, b_shape, b_nnz] =
      spblas::generate_csr<T, I>(m, n, nnz, 1);
  spblas::csr_wrapper<T, I> a(a_values, a_rowptr, a_colind, a_shape, a_nnz);
  spblas::csr_wrapper<T, I> b(b_values, b_rowptr, b_colind, b_shape, b_nnz);
  auto a_t = spblas::view::transpose(a);

  if constexpr (can_add<decltype(a_t), decltype(b), decltype(b)>) {
    auto b_dense = util::to_dense(b);
    auto expected = [&] {
      auto d = util::to_dense(a_t);
      for (std::size_t i = 0; i < d.size(); i++) {
        d[i] += b_dense[i];
      }
      return d;
    };

    // 2-stage
    spblas::csr_wrapper<T, I> c(m, n);
    spblas::operation_state state;
    spblas::add_compute(state, a_t, b, c);
    std::vector<I> c_rowptr(m + 1), c_colind(state.get_result_nnz());
    std::vector<T> c_values(state.get_result_nnz());
    c.update(c_values, c_rowptr, c_colind);
    spblas::add_fill(state, a_t, b, c);
    util::sort_output(c);
    EXPECT_EQ(util::to_dense(c), expected());

    // 3-stage, numeric again after A's values change.
    spblas::csr_wrapper<T, I> c3(m, n);
    spblas::add_compute_symbolic(state, a_t, b, c3);
    std::vector<I> rowptr3(m + 1), colind3(state.get_result_nnz());
    std::vector<T> values3(state.get_result_nnz());
    c3.update(values3, rowptr3, colind3);
    spblas::add_fill_symbolic(state, a_t, b, c3);
    for (auto&& v : a_values) {
      v *= T(2);
    }
    spblas::add_numeric(state, a_t, b, c3);
    util::sort_output(c3);
    EXPECT_EQ(util::to_dense(c3), expected());
  } else {
    GTEST_SKIP() << "this backend has no add for mixed orientations";
  }
}

TEST(Add, TransposedOperandIfSupported) {
  transposed_operand_add_test<double, spblas::index_t>();
}
