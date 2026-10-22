#include <gtest/gtest.h>

#include <cmath>
#include <complex>
#include <random>
#include <vector>

#include "util.hpp"
#include <spblas/spblas.hpp>

// __detail::analyze_view reduces stacks of transpose / conjugate / triangle /
// diagonal views to op(region(base)) (detail/view_analysis.hpp); the
// triangular solve accepts any such stack representing a triangular or
// diagonal matrix.

namespace {

using spblas::__detail::analyze_view;
using spblas::__detail::view_region;
namespace view = spblas::view;
using spblas::view::diag;
using spblas::view::uplo;

using T = std::complex<double>;
using I = spblas::index_t;

// A dense n x n matrix with a nonzero diagonal and its CSR / CSC arrays.
struct test_matrix {
  std::size_t n;
  std::vector<T> dense;
  std::vector<T> csr_values, csc_values;
  std::vector<I> rowptr, colind, colptr, rowind;

  explicit test_matrix(std::size_t n_, unsigned seed = 0)
      : n(n_), dense(n_ * n_, T(0)) {
    std::mt19937 g(seed);
    std::uniform_real_distribution<double> u(-1, 1);
    for (std::size_t i = 0; i < n; i++) {
      for (std::size_t j = 0; j < n; j++) {
        if (i == j) {
          dense[i * n + j] = T(4 + u(g), u(g));
        } else if (u(g) < -0.3) {
          dense[i * n + j] = T(u(g), u(g));
        }
      }
    }
    rowptr.push_back(0);
    for (std::size_t i = 0; i < n; i++) {
      for (std::size_t j = 0; j < n; j++) {
        if (dense[i * n + j] != T(0)) {
          csr_values.push_back(dense[i * n + j]);
          colind.push_back(I(j));
        }
      }
      rowptr.push_back(I(colind.size()));
    }
    colptr.push_back(0);
    for (std::size_t j = 0; j < n; j++) {
      for (std::size_t i = 0; i < n; i++) {
        if (dense[i * n + j] != T(0)) {
          csc_values.push_back(dense[i * n + j]);
          rowind.push_back(I(i));
        }
      }
      colptr.push_back(I(rowind.size()));
    }
  }

  spblas::csr_wrapper<T, I> csr() {
    return {csr_values, rowptr, colind, {I(n), I(n)}, I(colind.size())};
  }

  spblas::csc_wrapper<T, I> csc() {
    return {csc_values, colptr, rowind, {I(n), I(n)}, I(rowind.size())};
  }
};

// Dense counterparts of the views, applied to the represented matrix.
using dense_t = std::vector<T>;

dense_t d_transpose(const dense_t& m, std::size_t n) {
  dense_t r(n * n);
  for (std::size_t i = 0; i < n; i++) {
    for (std::size_t j = 0; j < n; j++) {
      r[j * n + i] = m[i * n + j];
    }
  }
  return r;
}

dense_t d_conjugate(dense_t m) {
  for (auto&& v : m) {
    v = std::conj(v);
  }
  return m;
}

void apply_diag(dense_t& r, const dense_t& m, std::size_t n, diag d) {
  for (std::size_t i = 0; i < n; i++) {
    r[i * n + i] = d == diag::explicit_diagonal ? m[i * n + i]
                   : d == diag::implicit_unit   ? T(1)
                                                : T(0);
  }
}

dense_t d_triangle(const dense_t& m, std::size_t n, uplo u,
                   diag d = diag::explicit_diagonal) {
  dense_t r(n * n, T(0));
  for (std::size_t i = 0; i < n; i++) {
    for (std::size_t j = 0; j < n; j++) {
      if (u == uplo::lower ? j < i : j > i) {
        r[i * n + j] = m[i * n + j];
      }
    }
  }
  apply_diag(r, m, n, d);
  return r;
}

dense_t d_diagonal(const dense_t& m, std::size_t n,
                   diag d = diag::explicit_diagonal) {
  dense_t r(n * n, T(0));
  apply_diag(r, m, n, d);
  return r;
}

// Solves m x = rhs by Gaussian elimination with partial pivoting.
std::vector<T> dense_solve(dense_t m, std::vector<T> rhs, std::size_t n) {
  for (std::size_t k = 0; k < n; k++) {
    std::size_t p = k;
    for (std::size_t i = k + 1; i < n; i++) {
      if (std::abs(m[i * n + k]) > std::abs(m[p * n + k])) {
        p = i;
      }
    }
    for (std::size_t j = 0; j < n; j++) {
      std::swap(m[k * n + j], m[p * n + j]);
    }
    std::swap(rhs[k], rhs[p]);
    for (std::size_t i = k + 1; i < n; i++) {
      T f = m[i * n + k] / m[k * n + k];
      for (std::size_t j = k; j < n; j++) {
        m[i * n + j] -= f * m[k * n + j];
      }
      rhs[i] -= f * rhs[k];
    }
  }
  std::vector<T> x(n);
  for (std::size_t i = n; i-- > 0;) {
    T s = rhs[i];
    for (std::size_t j = i + 1; j < n; j++) {
      s -= m[i * n + j] * x[j];
    }
    x[i] = s / m[i * n + i];
  }
  return x;
}

// Solves the view with triangular_solve and compares with a dense solve of
// `dense`, the matrix the view represents.
template <typename V>
void expect_solves(V&& v, const dense_t& dense, const char* what) {
  SCOPED_TRACE(what);
  const std::size_t n = std::size_t(std::lround(std::sqrt(dense.size())));
  std::vector<T> b(n);
  for (std::size_t i = 0; i < n; i++) {
    b[i] = T(double(i % 5) - 2, 1);
  }
  const T alpha(2, -1);
  std::vector<T> rhs(n);
  for (std::size_t i = 0; i < n; i++) {
    rhs[i] = alpha * b[i];
  }
  auto ref = dense_solve(dense, rhs, n);

  std::vector<T> x(n);
  spblas::operation_state state;
  spblas::triangular_solve(state, v, alpha, b, x);
  for (std::size_t i = 0; i < n; i++) {
    EXPECT_NEAR(std::abs(x[i] - ref[i]), 0, 1e-10 * (1 + std::abs(ref[i])))
        << "i = " << i;
  }
}

// Every spelling, for one base matrix (a wrapper or a matrix_opt).
template <typename A>
void solve_compositions(A a, const dense_t& d, std::size_t n) {
  const auto dt = d_transpose(d, n);

  expect_solves(view::triangle(a, uplo::lower), d_triangle(d, n, uplo::lower),
                "triangle(A, lower)");
  expect_solves(view::triangle(a, uplo::upper, diag::implicit_unit),
                d_triangle(d, n, uplo::upper, diag::implicit_unit),
                "triangle(A, upper, implicit_unit)");
  expect_solves(view::transpose(view::triangle(a, uplo::lower)),
                d_transpose(d_triangle(d, n, uplo::lower), n),
                "transpose(triangle(A, lower))");
  expect_solves(view::triangle(view::transpose(a), uplo::upper),
                d_triangle(dt, n, uplo::upper),
                "triangle(transpose(A), upper)");
  expect_solves(view::conjugate(view::transpose(
                    view::triangle(a, uplo::upper, diag::implicit_unit))),
                d_conjugate(d_transpose(
                    d_triangle(d, n, uplo::upper, diag::implicit_unit), n)),
                "conjugate(transpose(triangle(A, upper, implicit_unit)))");
  expect_solves(
      view::triangle(view::conjugate(view::transpose(a)), uplo::lower),
      d_triangle(d_conjugate(dt), n, uplo::lower),
      "triangle(conjugate(transpose(A)), lower)");
  expect_solves(view::triangle(view::triangle(a, uplo::lower), uplo::lower,
                               diag::implicit_unit),
                d_triangle(d_triangle(d, n, uplo::lower), n, uplo::lower,
                           diag::implicit_unit),
                "triangle(triangle(A, lower), lower, implicit_unit)");
  expect_solves(view::triangle(view::triangle(a, uplo::lower), uplo::upper),
                d_triangle(d_triangle(d, n, uplo::lower), n, uplo::upper),
                "triangle(triangle(A, lower), upper)");
  expect_solves(
      view::triangle(view::transpose(view::triangle(a, uplo::lower)),
                     uplo::lower),
      d_triangle(d_transpose(d_triangle(d, n, uplo::lower), n), n, uplo::lower),
      "triangle(transpose(triangle(A, lower)), lower)");
  expect_solves(view::diagonal(a), d_diagonal(d, n), "diagonal(A)");
  expect_solves(view::diagonal(view::conjugate(view::transpose(a))),
                d_diagonal(d_conjugate(dt), n),
                "diagonal(conjugate(transpose(A)))");
  expect_solves(view::diagonal(a, diag::implicit_unit),
                d_diagonal(d, n, diag::implicit_unit),
                "diagonal(A, implicit_unit)");
  expect_solves(
      view::conjugate(view::conjugate(view::triangle(a, uplo::lower))),
      d_triangle(d, n, uplo::lower),
      "conjugate(conjugate(triangle(A, lower)))");

  // Named (lvalue) views compose like temporaries.
  auto lower = view::triangle(a, uplo::lower);
  auto lower_t = view::transpose(lower);
  auto lower_th = view::conjugate(lower_t);
  expect_solves(lower_th,
                d_conjugate(d_transpose(d_triangle(d, n, uplo::lower), n)),
                "named: conjugate(transpose(triangle(A, lower)))");
}

} // namespace

TEST(ViewAnalysis, TransposeOfTriangleEqualsTriangleOfTranspose) {
  test_matrix m(6);
  spblas::matrix_opt a(m.csr());

  auto x = analyze_view(view::transpose(view::triangle(a, uplo::lower)));
  auto y = analyze_view(view::triangle(view::transpose(a), uplo::upper));
  static_assert(std::is_same_v<decltype(x), decltype(y)>);
  static_assert(decltype(x)::transpose && !decltype(x)::conjugate &&
                decltype(x)::structured);
  EXPECT_EQ(x.region, view_region::lower); // in A's coordinates
  EXPECT_EQ(y.region, view_region::lower);
  EXPECT_EQ(x.represented_region(), view_region::upper);

  // A CSR wrapper is transposed by reinterpreting it as CSC: the base differs,
  // the represented triangle does not.
  auto w = analyze_view(view::triangle(view::transpose(m.csr()), uplo::upper));
  static_assert(!decltype(w)::transpose);
  static_assert(spblas::__detail::is_csc_wrapper_v<decltype(w.base)>);
  EXPECT_EQ(w.represented_region(), view_region::upper);
}

TEST(ViewAnalysis, NamedViewsComposeLikeTemporaries) {
  test_matrix m(6);
  spblas::matrix_opt a(m.csr());

  auto t = view::transpose(a);
  auto tt = view::transpose(t);
  static_assert(!decltype(analyze_view(tt))::transpose);
  static_assert(spblas::__detail::is_matrix_opt_v<decltype(tt)>);

  auto c = view::conjugate(a);
  auto cc = view::conjugate(c);
  static_assert(!decltype(analyze_view(cc))::conjugate);

  auto l = view::triangle(a, uplo::lower);
  auto lu = view::triangle(l, uplo::upper);
  EXPECT_EQ(analyze_view(lu).region, view_region::diagonal);

  auto lt = view::transpose(l);
  auto ltl = view::triangle(lt, uplo::lower);
  EXPECT_EQ(analyze_view(ltl).region, view_region::diagonal);
}

TEST(ViewAnalysis, DiagonalHandling) {
  test_matrix m(6);
  spblas::matrix_opt a(m.csr());

  // An implied diagonal stays implied under explicit_diagonal; a later
  // implied diagonal replaces it.
  EXPECT_EQ(analyze_view(view::triangle(view::triangle(a, uplo::lower,
                                                       diag::implicit_unit),
                                        uplo::lower))
                .diag,
            diag::implicit_unit);
  EXPECT_EQ(analyze_view(view::triangle(view::triangle(a, uplo::lower,
                                                       diag::implicit_unit),
                                        uplo::lower, diag::implicit_zero))
                .diag,
            diag::implicit_zero);
  EXPECT_EQ(analyze_view(view::diagonal(view::triangle(a, uplo::upper),
                                        diag::implicit_unit))
                .diag,
            diag::implicit_unit);

  // The chains of PR #71's tests (Tag.LongChain).
  auto c1 = analyze_view(
      view::triangle(view::transpose(view::triangle(a, uplo::lower)),
                     uplo::upper, diag::implicit_zero));
  static_assert(decltype(c1)::transpose && !decltype(c1)::conjugate);
  EXPECT_EQ(c1.region, view_region::lower);
  EXPECT_EQ(c1.diag, diag::implicit_zero);

  auto c2 = analyze_view(view::conjugate(
      view::triangle(view::transpose(view::triangle(a, uplo::lower)),
                     uplo::lower, diag::implicit_zero)));
  static_assert(decltype(c2)::conjugate);
  EXPECT_EQ(c2.represented_region(), view_region::diagonal);
  EXPECT_EQ(c2.diag, diag::implicit_zero);
}

TEST(ViewAnalysis, UnstructuredStacks) {
  test_matrix m(6);
  spblas::matrix_opt a(m.csr());
  auto x = analyze_view(view::conjugate(view::transpose(a)));
  static_assert(!decltype(x)::structured);
  EXPECT_EQ(x.region, view_region::full);
  static_assert(!spblas::__detail::structured_view<decltype(a)>);
  static_assert(spblas::__detail::structured_view<decltype(view::transpose(
                    view::diagonal(a)))>);
}

// size(): the entries of the represented matrix, for any composition: the
// stored entries in the region, with an implied unit diagonal counted (one
// per diagonal position) and an implied zero diagonal not; computed once and
// kept by the view and its copies.
TEST(ViewAnalysis, StructuredViewSize) {
  test_matrix m(9, 1);
  const std::size_t n = m.n;
  auto stored = [&](auto&& in) {
    std::size_t count = 0;
    for (std::size_t i = 0; i < n; i++) {
      for (std::size_t j = 0; j < n; j++) {
        count += m.dense[i * n + j] != T(0) && in(i, j) ? 1 : 0;
      }
    }
    return count;
  };
  spblas::matrix_opt a(m.csr());
  const auto strictly_lower = stored([](auto i, auto j) { return j < i; });
  const auto diagonal = stored([](auto i, auto j) { return i == j; });

  EXPECT_EQ(view::triangle(a, uplo::lower).size(), strictly_lower + diagonal);
  EXPECT_EQ(view::triangle(a, uplo::lower, diag::implicit_unit).size(),
            strictly_lower + n);
  EXPECT_EQ(view::triangle(a, uplo::lower, diag::implicit_zero).size(),
            strictly_lower);
  // An implied diagonal stays implied under an outer explicit_diagonal.
  EXPECT_EQ(view::triangle(view::triangle(a, uplo::lower, diag::implicit_unit),
                           uplo::lower)
                .size(),
            strictly_lower + n);
  // upper(lower(A)^T) is the transpose of lower(A).
  EXPECT_EQ(view::triangle(view::transpose(view::triangle(a, uplo::lower)),
                           uplo::upper)
                .size(),
            strictly_lower + diagonal);
  EXPECT_EQ(
      view::triangle(view::triangle(m.csc(), uplo::lower), uplo::upper).size(),
      diagonal);
  EXPECT_EQ(view::diagonal(view::transpose(a)).size(), diagonal);
  EXPECT_EQ(view::diagonal(a, diag::implicit_unit).size(), n);
  EXPECT_EQ(view::diagonal(a, diag::implicit_zero).size(), 0u);
  EXPECT_EQ(view::triangle(view::transpose(view::triangle(a, uplo::upper)),
                           uplo::upper, diag::implicit_unit)
                .size(),
            n); // upper(upper(A)^T) is the diagonal

  auto t = view::triangle(m.csr(), uplo::upper);
  const auto upper = t.size();
  auto t_copy = t;
  // The pattern must not change while it is wrapped; doing so here shows
  // the size is not recounted.
  auto colind = m.colind;
  for (auto&& j : m.colind) {
    j = 0;
  }
  EXPECT_EQ(t.size(), upper);
  EXPECT_EQ(t_copy.size(), upper);
  m.colind = colind;
}

TEST(ViewAnalysis, TriangularSolveCompositions) {
  test_matrix m(17, 3);
  solve_compositions(m.csr(), m.dense, m.n);
  solve_compositions(m.csc(), m.dense, m.n);
  solve_compositions(spblas::matrix_opt(m.csr()), m.dense, m.n);
  solve_compositions(spblas::matrix_opt(m.csc()), m.dense, m.n);
}

TEST(ViewAnalysis, TriangularSolveRejectsImplicitZero) {
  test_matrix m(5);
  auto a = m.csr();
  std::vector<T> b(m.n, T(1)), x(m.n);
  spblas::operation_state state;
  EXPECT_THROW(
      spblas::triangular_solve(
          state,
          view::transpose(view::triangle(a, uplo::lower, diag::implicit_zero)),
          b, x),
      std::invalid_argument);
  EXPECT_THROW(spblas::triangular_solve(
                   state, view::diagonal(a, diag::implicit_zero), b, x),
               std::invalid_argument);
}

namespace {

template <typename M>
constexpr bool solvable =
    requires(spblas::operation_state& s, M m, std::vector<T>& v) {
      spblas::triangular_solve(s, m, v, v);
    };

} // namespace

TEST(ViewAnalysis, UnstructuredMatricesAreNotSolvable) {
  using A = spblas::csr_wrapper<T, I>;
  static_assert(!solvable<A>);
  static_assert(!solvable<decltype(view::transpose(std::declval<A>()))>);
  static_assert(solvable<decltype(view::transpose(
                    view::triangle(std::declval<A>(), uplo::lower)))>);
  static_assert(solvable<decltype(view::diagonal(std::declval<A>()))>);
}
