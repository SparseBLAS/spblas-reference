#include <gtest/gtest.h>

#include <complex>
#include <limits>

#include "util.hpp"
#include <spblas/spblas.hpp>

namespace {

template <typename T, typename I>
struct csr_matrix {
  std::vector<T> values;
  std::vector<I> rowptr;
  std::vector<I> colind;
  spblas::index<I> shape;
  I nnz;

  csr_matrix(std::size_t m, std::size_t n, std::size_t nnz_, std::size_t seed)
      : shape(m, n), nnz(nnz_) {
    auto [v, r, c, s, z] = spblas::generate_csr<T, I>(m, n, nnz_, seed);
    values = v;
    rowptr = r;
    colind = c;
  }

  spblas::csr_wrapper<T, I> csr() {
    return spblas::csr_wrapper<T, I>(values, rowptr, colind, shape, nnz);
  }
};

// The same matrix in CSC, built with spblas::transpose.
template <typename T, typename I>
struct csc_matrix {
  std::vector<T> values;
  std::vector<I> colptr;
  std::vector<I> rowind;
  spblas::index<I> shape;
  I nnz;

  explicit csc_matrix(csr_matrix<T, I>& a)
      : values(a.nnz), colptr(a.shape[1] + 1), rowind(a.nnz), shape(a.shape),
        nnz(a.nnz) {
    spblas::csr_wrapper<T, I> a_t(values, colptr, rowind,
                                  {a.shape[1], a.shape[0]}, a.nnz);
    spblas::operation_state state;
    spblas::transpose(state, a.csr(), a_t);
  }

  spblas::csc_wrapper<T, I> csc() {
    return spblas::csc_wrapper<T, I>(values, colptr, rowind, shape, nnz);
  }
};

template <typename T>
struct spgemm_reference {
  std::vector<T> values;          // dense alpha * A * B
  std::vector<double> magnitudes; // dense |alpha| * sum |a_ik| |b_kj|
  std::vector<bool> pattern;      // dense symbolic pattern
  std::size_t positions;          // size of the symbolic pattern

  // Summation order (and FMA contraction) differs between implementations,
  // so the error bound scales with the summands' magnitude, not the result's.
  void expect_matches(const std::vector<T>& actual) const {
    ASSERT_EQ(values.size(), actual.size());
    using R = decltype(std::abs(T{}));
    double eps = std::numeric_limits<R>::epsilon();
    for (std::size_t i = 0; i < values.size(); i++) {
      double tolerance = 64 * eps * magnitudes[i];
      if constexpr (spblas::__detail::is_std_complex_v<T>) {
        EXPECT_NEAR(values[i].real(), actual[i].real(), tolerance);
        EXPECT_NEAR(values[i].imag(), actual[i].imag(), tolerance);
      } else {
        EXPECT_NEAR(values[i], actual[i], tolerance);
      }
    }
  }
};

template <typename T, typename I, typename S>
spgemm_reference<T> spgemm_ref(csr_matrix<T, I>& a, csr_matrix<T, I>& b,
                               S alpha) {
  std::size_t m = a.shape[0], n = b.shape[1];
  spgemm_reference<T> r{std::vector<T>(m * n, T(0)),
                        std::vector<double>(m * n, 0.0),
                        std::vector<bool>(m * n, false), 0};
  auto& pattern = r.pattern;
  for (std::size_t i = 0; i < m; i++) {
    for (auto p = a.rowptr[i]; p < a.rowptr[i + 1]; p++) {
      auto k = a.colind[p];
      for (auto q = b.rowptr[k]; q < b.rowptr[k + 1]; q++) {
        auto ij = i * n + b.colind[q];
        r.values[ij] += a.values[p] * b.values[q];
        r.magnitudes[ij] += std::abs(a.values[p]) * std::abs(b.values[q]);
        pattern[ij] = true;
      }
    }
  }
  for (std::size_t ij = 0; ij < m * n; ij++) {
    r.values[ij] *= T(alpha);
    r.magnitudes[ij] *= std::abs(alpha);
  }
  r.positions = std::count(pattern.begin(), pattern.end(), true);
  return r;
}

// Runs compute -> allocate -> fill for any combination of inputs/output.
template <typename T, typename I, typename S, typename A, typename B,
          bool CscOutput>
auto run_spgemm(S alpha, A&& a, B&& b, std::size_t m, std::size_t n,
                std::bool_constant<CscOutput>) {
  struct result {
    std::vector<T> values;
    std::vector<I> ptr;
    std::vector<I> ind;
    std::size_t nnz;
  } r;

  spblas::operation_state state;
  auto run = [&](auto c) {
    spblas::multiply_inspect(state, alpha, a, b, c);
    spblas::multiply_compute(state, alpha, a, b, c);
    r.nnz = state.get_result_nnz();

    r.values.resize(r.nnz);
    r.ind.resize(r.nnz);
    r.ptr.resize((CscOutput ? n : m) + 1);
    c.update(r.values, r.ptr, r.ind);

    spblas::multiply_fill(state, alpha, a, b, c);
    return util::to_dense(c);
  };

  std::vector<T> dense;
  if constexpr (CscOutput) {
    dense = run(spblas::csc_wrapper<T, I>(m, n));
  } else {
    dense = run(spblas::csr_wrapper<T, I>(m, n));
  }
  return std::pair(r, dense);
}

template <bool CscA, bool CscB, bool CscC>
void spgemm_combination_test() {
  using T = float;
  using I = spblas::index_t;

  for (auto&& [m, k, nnz] : util::dims) {
    for (auto&& n : {m, k}) {
      for (T alpha : {T(1), T(-2)}) {
        csr_matrix<T, I> a_csr(m, k, nnz, 0);
        csr_matrix<T, I> b_csr(k, n, nnz, 1);
        csc_matrix<T, I> a_csc(a_csr);
        csc_matrix<T, I> b_csc(b_csr);

        auto a = [&] {
          if constexpr (CscA) {
            return a_csc.csc();
          } else {
            return a_csr.csr();
          }
        }();
        auto b = [&] {
          if constexpr (CscB) {
            return b_csc.csc();
          } else {
            return b_csr.csr();
          }
        }();

        auto [c, c_dense] =
            run_spgemm<T, I>(alpha, a, b, m, n, std::bool_constant<CscC>{});
        auto ref = spgemm_ref(a_csr, b_csr, alpha);

        EXPECT_EQ(c.nnz, ref.positions);
        EXPECT_EQ(c.ptr.back(), c.nnz);
        ref.expect_matches(c_dense);
      }
    }
  }
}

} // namespace

TEST(SpGEMM, CsrCsrCsr) {
  spgemm_combination_test<false, false, false>();
}

TEST(SpGEMM, CsrCsrCsc) {
  spgemm_combination_test<false, false, true>();
}

TEST(SpGEMM, CscCscCsc) {
  spgemm_combination_test<true, true, true>();
}

TEST(SpGEMM, CsrCscCsr) {
  spgemm_combination_test<false, true, false>();
}

TEST(SpGEMM, CsrCscCsc) {
  spgemm_combination_test<false, true, true>();
}

TEST(SpGEMM, CscCsrCsr) {
  spgemm_combination_test<true, false, false>();
}

TEST(SpGEMM, CscCsrCsc) {
  spgemm_combination_test<true, false, true>();
}

TEST(SpGEMM, AlphaZeroKeepsSymbolicPattern) {
  using T = float;
  using I = spblas::index_t;

  auto [m, k, nnz] = util::dims[2];
  csr_matrix<T, I> a(m, k, nnz, 0);
  csr_matrix<T, I> b(k, m, nnz, 1);

  auto [c, c_dense] =
      run_spgemm<T, I>(T(0), a.csr(), b.csr(), m, m, std::false_type{});
  auto ref = spgemm_ref(a, b, T(0));

  EXPECT_EQ(c.nnz, ref.positions);
  for (auto&& v : c.values) {
    EXPECT_EQ(v, T(0));
  }
}

TEST(SpGEMM, MatrixOptPolicyAndReuse) {
  using T = float;
  using I = spblas::index_t;

  for (auto&& [m, k, nnz] : util::dims) {
    csr_matrix<T, I> a_m(m, k, nnz, 0);
    csr_matrix<T, I> b_m(k, m, nnz, 1);
    spblas::matrix_opt a(a_m.csr());
    spblas::matrix_opt b(b_m.csr());
    auto ref = spgemm_ref(a_m, b_m, T(3));

    spblas::operation_state state;
    spblas::csr_wrapper<T, I> c(m, m);

    // Same state for two rounds: compute and fill again with the same arrays.
    std::vector<T> c_values;
    std::vector<I> c_rowptr, c_colind;
    for (int round = 0; round < 2; round++) {
      spblas::multiply_compute(util::test_policy{}, state, T(3), a, b, c);
      ASSERT_EQ(state.get_result_nnz(), ref.positions);

      c_values.assign(state.get_result_nnz(), T(-1));
      c_colind.assign(state.get_result_nnz(), 0);
      c_rowptr.assign(m + 1, 0);
      c.update(c_values, c_rowptr, c_colind);

      spblas::multiply_fill(util::test_policy{}, state, T(3), a, b, c);

      auto c_dense = util::to_dense(c);
      ref.expect_matches(c_dense);
    }
  }
}

TEST(SpGEMM, Conjugated) {
  using T = std::complex<float>;
  using I = spblas::index_t;

  for (auto&& [m, k, nnz] : util::dims) {
    csr_matrix<T, I> a(m, k, nnz, 0);
    csr_matrix<T, I> b(k, m, nnz, 1);

    csr_matrix<T, I> a_conj = a;
    for (auto&& v : a_conj.values) {
      v = std::conj(v);
    }
    auto ref = spgemm_ref(a_conj, b, T(1));

    auto [c, c_dense] = run_spgemm<T, I>(T(1), spblas::view::conjugate(a.csr()),
                                         b.csr(), m, m, std::false_type{});

    EXPECT_EQ(c.nnz, ref.positions);
    ref.expect_matches(c_dense);
  }
}

namespace {

// Dense pattern of a CSR (by_columns = false) or CSC (true) pattern.
template <typename I>
std::vector<bool> dense_pattern(const std::vector<I>& pointers,
                                const std::vector<I>& indices, std::size_t m,
                                std::size_t n, bool by_columns) {
  std::vector<bool> pattern(m * n, false);
  for (std::size_t o = 0; o + 1 < pointers.size(); o++) {
    for (auto p = pointers[o]; p < pointers[o + 1]; p++) {
      std::size_t i = by_columns ? indices[p] : o;
      std::size_t j = by_columns ? o : indices[p];
      EXPECT_FALSE(pattern[i * n + j]) << "duplicate entry";
      pattern[i * n + j] = true;
    }
  }
  return pattern;
}

template <bool CscA, bool CscB, bool CscC>
void spgemm_three_stage_test() {
  using T = float;
  using I = spblas::index_t;
  using C = std::conditional_t<CscC, spblas::csc_wrapper<T, I>,
                               spblas::csr_wrapper<T, I>>;

  for (auto&& [m, k, nnz] : util::dims) {
    for (auto&& n : {m, k}) {
      csr_matrix<T, I> a_csr(m, k, nnz, 0);
      csr_matrix<T, I> b_csr(k, n, nnz, 1);
      csc_matrix<T, I> a_csc(a_csr);
      csc_matrix<T, I> b_csc(b_csr);
      auto a = [&] {
        if constexpr (CscA) {
          return a_csc.csc();
        } else {
          return a_csr.csr();
        }
      }();
      auto b = [&] {
        if constexpr (CscB) {
          return b_csc.csc();
        } else {
          return b_csr.csr();
        }
      }();

      T alpha = -2;
      auto ref = spgemm_ref(a_csr, b_csr, alpha);

      spblas::operation_state state;
      C c(m, n);
      spblas::multiply_inspect(state, alpha, a, b, c);
      spblas::multiply_compute_symbolic(state, alpha, a, b, c);
      ASSERT_EQ(state.get_result_nnz(), ref.positions);

      // Pattern first; the values array is attached only afterwards.
      std::vector<I> pointers((CscC ? n : m) + 1);
      std::vector<I> indices(state.get_result_nnz());
      std::vector<T> values;
      c.update(values, pointers, indices);
      spblas::multiply_fill_symbolic(state, alpha, a, b, c);

      EXPECT_EQ(pointers.back(), state.get_result_nnz());
      EXPECT_EQ(dense_pattern(pointers, indices, m, n, CscC), ref.pattern);

      values.assign(state.get_result_nnz(), T(-1));
      c.update(values, pointers, indices);
      spblas::multiply_numeric(state, alpha, a, b, c);
      ref.expect_matches(util::to_dense(c));

      // New values for A, same pattern: only the numeric stage reruns.
      for (auto&& v : a_csr.values) {
        v *= T(3);
      }
      for (auto&& v : a_csc.values) {
        v *= T(3);
      }
      spblas::multiply_numeric(state, alpha, a, b, c);
      spgemm_ref(a_csr, b_csr, alpha).expect_matches(util::to_dense(c));
    }
  }
}

} // namespace

TEST(SpGEMM, ThreeStageCsrCsrCsr) {
  spgemm_three_stage_test<false, false, false>();
}

TEST(SpGEMM, ThreeStageCscCscCsc) {
  spgemm_three_stage_test<true, true, true>();
}

TEST(SpGEMM, ThreeStageCsrCscCsr) {
  spgemm_three_stage_test<false, true, false>();
}

TEST(SpGEMM, ThreeStageCscCsrCsc) {
  spgemm_three_stage_test<true, false, true>();
}

TEST(SpGEMM, ThreeStageMatrixOptAndPolicy) {
  using T = double;
  using I = spblas::index_t;

  auto [m, k, nnz] = util::dims[1];
  csr_matrix<T, I> a_m(m, k, nnz, 0);
  csr_matrix<T, I> b_m(k, m, nnz, 1);
  spblas::matrix_opt a(a_m.csr());
  spblas::matrix_opt b(b_m.csr());
  auto ref = spgemm_ref(a_m, b_m, T(0.5));

  spblas::operation_state state;
  spblas::csr_wrapper<T, I> c(m, m);
  spblas::multiply_compute_symbolic(util::test_policy{}, state, T(0.5), a, b,
                                    c);
  std::vector<I> rowptr(m + 1), colind(state.get_result_nnz());
  std::vector<T> values(state.get_result_nnz());
  c.update(values, rowptr, colind);
  spblas::multiply_fill_symbolic(util::test_policy{}, state, T(0.5), a, b, c);
  // fill_symbolic leaves an already-allocated values array untouched.
  EXPECT_TRUE(
      std::all_of(values.begin(), values.end(), [](T v) { return v == T(0); }));
  spblas::multiply_numeric(util::test_policy{}, state, T(0.5), a, b, c);

  EXPECT_EQ(dense_pattern(rowptr, colind, m, m, false), ref.pattern);
  ref.expect_matches(util::to_dense(c));
}

TEST(SpGEMM, ThreeStageErrors) {
  using T = float;
  using I = spblas::index_t;

  auto [m, k, nnz] = util::dims[0];
  csr_matrix<T, I> a(m, k, nnz, 0);
  csr_matrix<T, I> b(k, m, nnz, 1);
  csr_matrix<T, I> other_b(k, m, nnz, 2);

  spblas::operation_state state;
  spblas::csr_wrapper<T, I> c(m, m);
  spblas::multiply_compute_symbolic(state, T(1), a.csr(), b.csr(), c);

  // The pointer array must be attached before filling the pattern.
  std::vector<I> colind(state.get_result_nnz());
  std::vector<T> no_values;
  std::vector<I> no_rowptr;
  c.update(no_values, no_rowptr, colind);
  EXPECT_THROW(spblas::multiply_fill_symbolic(state, T(1), a.csr(), b.csr(), c),
               std::invalid_argument);

  std::vector<I> rowptr(m + 1);
  c.update(no_values, rowptr, colind);
  spblas::multiply_fill_symbolic(state, T(1), a.csr(), b.csr(), c);

  // The values array must be allocated before the numeric stage.
  EXPECT_THROW(spblas::multiply_numeric(state, T(1), a.csr(), b.csr(), c),
               std::invalid_argument);

  // A pattern computed for A * B does not hold A * other_B.
  std::vector<T> values(state.get_result_nnz());
  c.update(values, rowptr, colind);
  EXPECT_THROW(spblas::multiply_numeric(state, T(1), a.csr(), other_b.csr(), c),
               std::invalid_argument);
}
