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
                        std::vector<double>(m * n, 0.0), 0};
  std::vector<bool> pattern(m * n, false);
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
