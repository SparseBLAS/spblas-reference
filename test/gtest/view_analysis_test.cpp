#include <gtest/gtest.h>

#include <cmath>
#include <complex>
#include <random>
#include <type_traits>
#include <utility>
#include <vector>

#include "util.hpp"
#include <spblas/spblas.hpp>

// __detail::analyze_view reduces stacks of transpose / conjugate / triangle /
// diagonal / symmetric / hermitian views to op(structure(region(base))) at
// compile time (detail/view_analysis.hpp). The triangular solve accepts any
// stack representing a triangular or diagonal matrix; SpMV / SpMM accept any
// structured stack.

namespace {

using spblas::__detail::describe_view;
using spblas::__detail::diag_kind;
using spblas::__detail::view_region;
using spblas::__detail::view_structure;
namespace view = spblas::view;
namespace uplo = spblas::view::uplo;
namespace diag = spblas::view::diag;

using T = std::complex<double>;
using I = spblas::index_t;

// The analysis of a view expression (unevaluated: only its type is used).
#define DESC(...) describe_view<decltype(__VA_ARGS__)>()

// A dense n x n matrix with a nonzero (complex) diagonal and its CSR / CSC
// arrays.
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

//
// Dense counterparts of the views, applied to the represented matrix.
//
using dense_t = std::vector<T>;

std::size_t dim(const dense_t& m) {
  return std::size_t(std::lround(std::sqrt(m.size())));
}

dense_t d_transpose(const dense_t& m) {
  const std::size_t n = dim(m);
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

template <typename D>
T diagonal_value(const dense_t& m, std::size_t i, D) {
  if constexpr (std::is_same_v<D, diag::explicit_diagonal_t>) {
    return m[i * dim(m) + i];
  } else if constexpr (std::is_same_v<D, diag::implicit_unit_t>) {
    return T(1);
  } else {
    return T(0);
  }
}

template <typename U, typename D = diag::explicit_diagonal_t>
dense_t d_triangle(const dense_t& m, U, D d = {}) {
  const std::size_t n = dim(m);
  constexpr bool lower = std::is_same_v<U, uplo::lower_t>;
  dense_t r(n * n, T(0));
  for (std::size_t i = 0; i < n; i++) {
    for (std::size_t j = 0; j < n; j++) {
      if (lower ? j < i : j > i) {
        r[i * n + j] = m[i * n + j];
      }
    }
    r[i * n + i] = diagonal_value(m, i, d);
  }
  return r;
}

template <typename D = diag::explicit_diagonal_t>
dense_t d_diagonal(const dense_t& m, D d = {}) {
  const std::size_t n = dim(m);
  dense_t r(n * n, T(0));
  for (std::size_t i = 0; i < n; i++) {
    r[i * n + i] = diagonal_value(m, i, d);
  }
  return r;
}

// T + S^T (symmetric) or T + S^H with the real part of the diagonal
// (hermitian), T the u triangle of m.
template <bool Hermitian, typename U, typename D>
dense_t d_complete(const dense_t& m, U u, D d) {
  const std::size_t n = dim(m);
  dense_t t = d_triangle(m, u, d);
  dense_t r = t;
  for (std::size_t i = 0; i < n; i++) {
    for (std::size_t j = 0; j < n; j++) {
      if (i != j && t[i * n + j] != T(0)) {
        r[j * n + i] = Hermitian ? std::conj(t[i * n + j]) : t[i * n + j];
      }
    }
    if (Hermitian) {
      r[i * n + i] = T(r[i * n + i].real());
    }
  }
  return r;
}

template <typename U, typename D = diag::explicit_diagonal_t>
dense_t d_symmetric(const dense_t& m, U u, D d = {}) {
  return d_complete<false>(m, u, d);
}

template <typename U, typename D = diag::explicit_diagonal_t>
dense_t d_hermitian(const dense_t& m, U u, D d = {}) {
  return d_complete<true>(m, u, d);
}

// Solves m x = rhs by Gaussian elimination with partial pivoting.
std::vector<T> dense_solve(dense_t m, std::vector<T> rhs) {
  const std::size_t n = dim(m);
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

std::vector<T> test_vector(std::size_t n, int shift) {
  std::vector<T> v(n);
  for (std::size_t i = 0; i < n; i++) {
    v[i] = T(double((i + shift) % 5) - 2, double(i % 3) - 1);
  }
  return v;
}

void expect_near(const std::vector<T>& expected, const std::vector<T>& actual) {
  ASSERT_EQ(expected.size(), actual.size());
  for (std::size_t i = 0; i < expected.size(); i++) {
    EXPECT_NEAR(std::abs(expected[i] - actual[i]), 0,
                1e-10 * (1 + std::abs(expected[i])))
        << "i = " << i;
  }
}

//
// Triangular solve
//

// Solves with the view and compares with a dense solve of `dense`, the
// matrix the view represents.
template <typename V>
void expect_solves(const V& v, const dense_t& dense, const char* what) {
  SCOPED_TRACE(what);
  const std::size_t n = dim(dense);
  const T alpha(2, -1);
  auto b = test_vector(n, 0);
  std::vector<T> rhs(n);
  for (std::size_t i = 0; i < n; i++) {
    rhs[i] = alpha * b[i];
  }

  std::vector<T> x(n);
  spblas::operation_state state;
  spblas::triangular_solve(state, v, alpha, b, x);
  expect_near(dense_solve(dense, rhs), x);
}

template <typename A>
void solve_compositions(A a, const dense_t& d) {
  const auto dt = d_transpose(d);

  expect_solves(view::triangle(a, uplo::lower), d_triangle(d, uplo::lower),
                "triangle(A, lower)");
  expect_solves(view::triangle(a, uplo::upper, diag::implicit_unit),
                d_triangle(d, uplo::upper, diag::implicit_unit),
                "triangle(A, upper, implicit_unit)");
  expect_solves(view::transpose(view::triangle(a, uplo::lower)),
                d_transpose(d_triangle(d, uplo::lower)),
                "transpose(triangle(A, lower))");
  expect_solves(view::triangle(view::transpose(a), uplo::upper),
                d_triangle(dt, uplo::upper), "triangle(transpose(A), upper)");
  expect_solves(
      view::conjugate(
          view::transpose(view::triangle(a, uplo::upper, diag::implicit_unit))),
      d_conjugate(d_transpose(d_triangle(d, uplo::upper, diag::implicit_unit))),
      "conjugate(transpose(triangle(A, upper, implicit_unit)))");
  expect_solves(
      view::triangle(view::conjugate(view::transpose(a)), uplo::lower),
      d_triangle(d_conjugate(dt), uplo::lower),
      "triangle(conjugate(transpose(A)), lower)");
  expect_solves(
      view::triangle(view::triangle(a, uplo::lower), uplo::lower,
                     diag::implicit_unit),
      d_triangle(d_triangle(d, uplo::lower), uplo::lower, diag::implicit_unit),
      "triangle(triangle(A, lower), lower, implicit_unit)");
  expect_solves(view::triangle(view::triangle(a, uplo::lower), uplo::upper),
                d_triangle(d_triangle(d, uplo::lower), uplo::upper),
                "triangle(triangle(A, lower), upper)");
  expect_solves(
      view::triangle(view::transpose(view::triangle(a, uplo::lower)),
                     uplo::lower),
      d_triangle(d_transpose(d_triangle(d, uplo::lower)), uplo::lower),
      "triangle(transpose(triangle(A, lower)), lower)");
  expect_solves(view::diagonal(a), d_diagonal(d), "diagonal(A)");
  expect_solves(view::diagonal(view::conjugate(view::transpose(a))),
                d_diagonal(d_conjugate(dt)),
                "diagonal(conjugate(transpose(A)))");
  expect_solves(view::diagonal(a, diag::implicit_unit),
                d_diagonal(d, diag::implicit_unit),
                "diagonal(A, implicit_unit)");
  expect_solves(
      view::conjugate(view::conjugate(view::triangle(a, uplo::lower))),
      d_triangle(d, uplo::lower), "conjugate(conjugate(triangle(A, lower)))");
  // Triangles of symmetric / hermitian matrices.
  expect_solves(view::triangle(view::symmetric(a, uplo::lower), uplo::upper),
                d_triangle(d_symmetric(d, uplo::lower), uplo::upper),
                "triangle(symmetric(A, lower), upper)");
  expect_solves(
      view::triangle(view::hermitian(a, uplo::lower), uplo::upper,
                     diag::implicit_unit),
      d_triangle(d_hermitian(d, uplo::lower), uplo::upper, diag::implicit_unit),
      "triangle(hermitian(A, lower), upper, implicit_unit)");

  // Named (lvalue) views compose like temporaries.
  auto lower = view::triangle(a, uplo::lower);
  auto lower_t = view::transpose(lower);
  auto lower_h = view::conjugate(lower_t);
  expect_solves(lower_h, d_conjugate(d_transpose(d_triangle(d, uplo::lower))),
                "named: conjugate(transpose(triangle(A, lower)))");

  // A triangle chosen at run time.
  for (bool is_lower : {true, false}) {
    view::with_uplo(is_lower, [&](auto u) {
      expect_solves(view::triangle(a, u), d_triangle(d, u),
                    "with_uplo: triangle(A, u)");
    });
  }
}

//
// SpMV / SpMM
//

// y = alpha * A * x + beta * z with the view, compared with `dense`.
template <typename V>
void expect_spmv(const V& v, const dense_t& dense, const char* what) {
  SCOPED_TRACE(what);
  const std::size_t n = dim(dense);
  const T alpha(1.5, -0.5), beta(-1, 2);
  auto x = test_vector(n, 1);
  auto z = test_vector(n, 3);

  std::vector<T> ref(n);
  for (std::size_t i = 0; i < n; i++) {
    T s = 0;
    for (std::size_t j = 0; j < n; j++) {
      s += dense[i * n + j] * x[j];
    }
    ref[i] = alpha * s + beta * z[i];
  }

  std::vector<T> y(n);
  spblas::operation_state state;
  spblas::multiply(state, alpha, v, x, beta, z, y);
  expect_near(ref, y);

  // Without beta / z, and in place (z aliases y).
  std::vector<T> y2(n);
  spblas::multiply(state, T(1), v, x, y2);
  std::vector<T> ref2(n, T(0));
  for (std::size_t i = 0; i < n; i++) {
    for (std::size_t j = 0; j < n; j++) {
      ref2[i] += dense[i * n + j] * x[j];
    }
  }
  expect_near(ref2, y2);
}

template <typename V>
void expect_spmm(const V& v, const dense_t& dense, const char* what) {
  SCOPED_TRACE(what);
  const std::size_t n = dim(dense), k = 3;
  const T alpha(0.5, 1);
  std::vector<T> x_values(n * k), y_values(n * k, T(0)), ref(n * k, T(0));
  for (std::size_t i = 0; i < n * k; i++) {
    x_values[i] = T(double(i % 7) - 3, double(i % 2));
  }
  for (std::size_t i = 0; i < n; i++) {
    for (std::size_t c = 0; c < k; c++) {
      for (std::size_t j = 0; j < n; j++) {
        ref[i * k + c] += alpha * dense[i * n + j] * x_values[j * k + c];
      }
    }
  }
  spblas::mdspan_row_major<T, I> x(x_values.data(), n, k);
  spblas::mdspan_row_major<T, I> y(y_values.data(), n, k);
  spblas::operation_state state;
  spblas::multiply(state, alpha, v, x, y);
  expect_near(ref, y_values);
}

template <typename A>
void multiply_compositions(A a, const dense_t& d) {
  const auto dt = d_transpose(d);

  // SpTRMV
  expect_spmv(view::triangle(a, uplo::lower), d_triangle(d, uplo::lower),
              "triangle(A, lower)");
  expect_spmv(view::triangle(a, uplo::upper, diag::implicit_unit),
              d_triangle(d, uplo::upper, diag::implicit_unit),
              "triangle(A, upper, implicit_unit)");
  expect_spmv(view::transpose(view::triangle(a, uplo::lower)),
              d_transpose(d_triangle(d, uplo::lower)),
              "transpose(triangle(A, lower))");
  expect_spmv(view::triangle(a, uplo::lower, diag::implicit_zero),
              d_triangle(d, uplo::lower, diag::implicit_zero),
              "triangle(A, lower, implicit_zero)");
  // Diagonal
  expect_spmv(view::diagonal(a), d_diagonal(d), "diagonal(A)");
  expect_spmv(view::diagonal(a, diag::implicit_unit),
              d_diagonal(d, diag::implicit_unit), "diagonal(A, implicit_unit)");
  // SpSYMV
  expect_spmv(view::symmetric(a, uplo::lower), d_symmetric(d, uplo::lower),
              "symmetric(A, lower)");
  expect_spmv(view::symmetric(a, uplo::upper, diag::implicit_unit),
              d_symmetric(d, uplo::upper, diag::implicit_unit),
              "symmetric(A, upper, implicit_unit)");
  expect_spmv(view::transpose(view::symmetric(a, uplo::upper)),
              d_transpose(d_symmetric(d, uplo::upper)),
              "transpose(symmetric(A, upper))");
  expect_spmv(view::symmetric(view::transpose(a), uplo::upper),
              d_symmetric(dt, uplo::upper), "symmetric(transpose(A), upper)");
  expect_spmv(view::conjugate(view::symmetric(a, uplo::lower)),
              d_conjugate(d_symmetric(d, uplo::lower)),
              "conjugate(symmetric(A, lower))");
  expect_spmv(view::symmetric(view::triangle(a, uplo::lower), uplo::upper),
              d_symmetric(d_triangle(d, uplo::lower), uplo::upper),
              "symmetric(triangle(A, lower), upper)");
  expect_spmv(view::symmetric(view::symmetric(a, uplo::lower), uplo::upper),
              d_symmetric(d_symmetric(d, uplo::lower), uplo::upper),
              "symmetric(symmetric(A, lower), upper)");
  // SpHEMV (A's diagonal is complex: only its real part is used)
  expect_spmv(view::hermitian(a, uplo::lower), d_hermitian(d, uplo::lower),
              "hermitian(A, lower)");
  expect_spmv(view::hermitian(a, uplo::upper, diag::implicit_unit),
              d_hermitian(d, uplo::upper, diag::implicit_unit),
              "hermitian(A, upper, implicit_unit)");
  expect_spmv(view::transpose(view::hermitian(a, uplo::lower)),
              d_transpose(d_hermitian(d, uplo::lower)),
              "transpose(hermitian(A, lower))");
  expect_spmv(view::conjugate(view::transpose(view::hermitian(a, uplo::upper))),
              d_conjugate(d_transpose(d_hermitian(d, uplo::upper))),
              "conjugate(transpose(hermitian(A, upper)))");
  expect_spmv(view::hermitian(view::transpose(a), uplo::lower),
              d_hermitian(dt, uplo::lower), "hermitian(transpose(A), lower)");
  expect_spmv(view::hermitian(view::conjugate(view::transpose(a)), uplo::upper),
              d_hermitian(d_conjugate(dt), uplo::upper),
              "hermitian(conjugate(transpose(A)), upper)");
  // A triangle of a symmetric / hermitian matrix.
  expect_spmv(view::triangle(view::symmetric(a, uplo::lower), uplo::upper),
              d_triangle(d_symmetric(d, uplo::lower), uplo::upper),
              "triangle(symmetric(A, lower), upper)");
  expect_spmv(
      view::triangle(view::hermitian(a, uplo::upper), uplo::lower,
                     diag::implicit_unit),
      d_triangle(d_hermitian(d, uplo::upper), uplo::lower, diag::implicit_unit),
      "triangle(hermitian(A, upper), lower, implicit_unit)");
  // Named views.
  auto s = view::symmetric(a, uplo::lower);
  auto s_t = view::transpose(s);
  expect_spmv(s_t, d_symmetric(d, uplo::lower),
              "named: transpose(symmetric(A, lower))");

  // SpMM
  expect_spmm(view::triangle(a, uplo::upper, diag::implicit_unit),
              d_triangle(d, uplo::upper, diag::implicit_unit),
              "SpMM triangle(A, upper, implicit_unit)");
  expect_spmm(view::symmetric(a, uplo::lower), d_symmetric(d, uplo::lower),
              "SpMM symmetric(A, lower)");
  expect_spmm(view::hermitian(view::transpose(a), uplo::upper),
              d_hermitian(dt, uplo::upper),
              "SpMM hermitian(transpose(A), upper)");
}

} // namespace

TEST(ViewAnalysis, TransposeOfTriangleEqualsTriangleOfTranspose) {
  test_matrix m(6);
  spblas::matrix_opt a(m.csr());

  constexpr auto x = DESC(view::transpose(view::triangle(a, uplo::lower)));
  constexpr auto y = DESC(view::triangle(view::transpose(a), uplo::upper));
  static_assert(x == y);
  static_assert(x.transpose && !x.conjugate && x.structured);
  static_assert(x.region == view_region::lower); // in A's coordinates

  // A CSR wrapper is transposed by reinterpreting it as CSC: the base
  // differs, the represented triangle does not.
  auto w = spblas::__detail::analyze_view(
      view::triangle(view::transpose(m.csr()), uplo::upper));
  static_assert(!decltype(w)::transpose);
  static_assert(spblas::__detail::is_csc_wrapper_v<decltype(w.base)>);
  static_assert(decltype(w)::represented_region == view_region::upper);
}

TEST(ViewAnalysis, NamedViewsComposeLikeTemporaries) {
  test_matrix m(6);
  spblas::matrix_opt a(m.csr());

  auto t = view::transpose(a);
  auto tt = view::transpose(t);
  static_assert(spblas::__detail::is_matrix_opt_v<decltype(tt)>);

  auto c = view::conjugate(a);
  auto cc = view::conjugate(c);
  static_assert(spblas::__detail::is_matrix_opt_v<decltype(cc)>);

  auto l = view::triangle(a, uplo::lower);
  auto lu = view::triangle(l, uplo::upper);
  static_assert(DESC(lu).region == view_region::diagonal);
  static_assert(DESC(lu) == DESC(view::diagonal(a)));

  auto lt = view::transpose(l);
  auto ltl = view::triangle(lt, uplo::lower);
  static_assert(DESC(ltl) == DESC(view::diagonal(a)));
}

TEST(ViewAnalysis, DiagonalHandling) {
  test_matrix m(6);
  spblas::matrix_opt a(m.csr());

  // An implied diagonal stays implied under explicit_diagonal; a later
  // implied diagonal replaces it.
  static_assert(
      DESC(view::triangle(view::triangle(a, uplo::lower, diag::implicit_unit),
                          uplo::lower))
          .diag == diag_kind::implicit_unit);
  static_assert(
      DESC(view::triangle(view::triangle(a, uplo::lower, diag::implicit_unit),
                          uplo::lower, diag::implicit_zero))
          .diag == diag_kind::implicit_zero);
  static_assert(
      DESC(view::diagonal(view::triangle(a, uplo::upper), diag::implicit_unit))
          .diag == diag_kind::implicit_unit);

  // The chains of PR #71's tests (Tag.LongChain).
  constexpr auto c1 =
      DESC(view::triangle(view::transpose(view::triangle(a, uplo::lower)),
                          uplo::upper, diag::implicit_zero));
  static_assert(c1.transpose && !c1.conjugate);
  static_assert(c1.region == view_region::lower);
  static_assert(c1.diag == diag_kind::implicit_zero);

  constexpr auto c2 = DESC(view::conjugate(
      view::triangle(view::transpose(view::triangle(a, uplo::lower)),
                     uplo::lower, diag::implicit_zero)));
  static_assert(c2.conjugate && !c2.transpose); // a diagonal drops transpose
  static_assert(c2.region == view_region::diagonal);
  static_assert(c2.diag == diag_kind::implicit_zero);
}

// The composition rules of symmetric / hermitian views (plan §8).
TEST(ViewAnalysis, SymmetricAndHermitianRules) {
  test_matrix m(6);
  spblas::matrix_opt a(m.csr());

  constexpr auto sym_lower = DESC(view::symmetric(a, uplo::lower));
  static_assert(sym_lower.structure == view_structure::symmetric &&
                sym_lower.region == view_region::lower &&
                !sym_lower.transpose && !sym_lower.conjugate);
  // S^T == S
  static_assert(DESC(view::transpose(view::symmetric(a, uplo::lower))) ==
                sym_lower);
  // symmetric(A^T, u) == symmetric(A, flip(u))
  static_assert(DESC(view::symmetric(view::transpose(a), uplo::upper)) ==
                sym_lower);
  // triangle(S(A, u), u) == triangle(A, u); the other triangle is its
  // transpose
  static_assert(
      DESC(view::triangle(view::symmetric(a, uplo::lower), uplo::lower)) ==
      DESC(view::triangle(a, uplo::lower)));
  static_assert(
      DESC(view::triangle(view::symmetric(a, uplo::lower), uplo::upper)) ==
      DESC(view::transpose(view::triangle(a, uplo::lower))));
  // symmetric(triangle(A, u), u) == symmetric(A, u); other u: the diagonal
  static_assert(DESC(view::symmetric(view::triangle(a, uplo::lower),
                                     uplo::lower)) == sym_lower);
  static_assert(DESC(view::symmetric(view::triangle(a, uplo::lower),
                                     uplo::upper)) == DESC(view::diagonal(a)));
  // S(S(A, u), v) == S(A, u); diagonal(S(A, u)) == diagonal(A)
  static_assert(DESC(view::symmetric(view::symmetric(a, uplo::lower),
                                     uplo::upper)) == sym_lower);
  static_assert(DESC(view::diagonal(view::symmetric(a, uplo::lower))) ==
                DESC(view::diagonal(a)));

  constexpr auto herm_upper = DESC(view::hermitian(a, uplo::upper));
  static_assert(herm_upper.structure == view_structure::hermitian);
  // H^T == conj(H); H^H == H
  static_assert(DESC(view::transpose(view::hermitian(a, uplo::upper))) ==
                DESC(view::conjugate(view::hermitian(a, uplo::upper))));
  static_assert(DESC(view::conjugate(view::transpose(
                    view::hermitian(a, uplo::upper)))) == herm_upper);
  // hermitian(A^H, u) == hermitian(A, flip(u))
  static_assert(DESC(view::hermitian(view::conjugate(view::transpose(a)),
                                     uplo::lower)) == herm_upper);
  // The other triangle of a hermitian matrix is the conjugate transpose (with
  // an implied diagonal: the explicit one would need its real part).
  static_assert(DESC(view::triangle(view::hermitian(a, uplo::upper),
                                    uplo::lower, diag::implicit_unit)) ==
                DESC(view::conjugate(view::transpose(
                    view::triangle(a, uplo::upper, diag::implicit_unit)))));

  // Hermitian of real values is symmetric.
  using R = spblas::csr_wrapper<double, I>;
  static_assert(describe_view<decltype(view::hermitian(std::declval<R>(),
                                                       uplo::lower))>() ==
                describe_view<decltype(view::symmetric(std::declval<R>(),
                                                       uplo::lower))>());

  // Not representable: any operation (even asking for size()) on these is a
  // static_assert.
  static_assert(
      !DESC(view::triangle(view::hermitian(a, uplo::lower), uplo::lower))
           .valid);
  static_assert(!DESC(view::diagonal(view::hermitian(a, uplo::lower))).valid);
  static_assert(
      !DESC(view::symmetric(view::hermitian(a, uplo::lower), uplo::lower))
           .valid);
}

TEST(ViewAnalysis, UnstructuredStacks) {
  test_matrix m(6);
  spblas::matrix_opt a(m.csr());
  constexpr auto x = DESC(view::conjugate(view::transpose(a)));
  static_assert(!x.structured && x.region == view_region::full);
  static_assert(!spblas::__detail::structured_view<decltype(a)>);
  static_assert(spblas::__detail::structured_view<decltype(view::transpose(
                    view::diagonal(a)))>);
}

// size(): the entries of the represented matrix, for any composition: the
// stored entries in the region (strict-triangle entries twice for symmetric /
// hermitian views), an implied unit diagonal counted (one per diagonal
// position), an implied zero diagonal not; computed once and kept by the
// view and its copies.
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
  const auto strictly_upper = stored([](auto i, auto j) { return j > i; });
  const auto diagonal = stored([](auto i, auto j) { return i == j; });

  EXPECT_EQ(view::triangle(a, uplo::lower).size(), strictly_lower + diagonal);
  EXPECT_EQ(view::triangle(a, uplo::lower, diag::implicit_unit).size(),
            strictly_lower + n);
  EXPECT_EQ(view::triangle(a, uplo::lower, diag::implicit_zero).size(),
            strictly_lower);
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

  // Symmetric / hermitian: the represented (mirrored) entries.
  EXPECT_EQ(view::symmetric(a, uplo::lower).size(),
            2 * strictly_lower + diagonal);
  EXPECT_EQ(view::hermitian(a, uplo::upper, diag::implicit_unit).size(),
            2 * strictly_upper + n);
  EXPECT_EQ(view::symmetric(m.csc(), uplo::upper, diag::implicit_zero).size(),
            2 * strictly_upper);
  EXPECT_EQ(view::triangle(view::symmetric(a, uplo::lower), uplo::upper).size(),
            strictly_lower + diagonal);

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
  solve_compositions(m.csr(), m.dense);
  solve_compositions(m.csc(), m.dense);
  solve_compositions(spblas::matrix_opt(m.csr()), m.dense);
  solve_compositions(spblas::matrix_opt(m.csc()), m.dense);
}

TEST(ViewAnalysis, StructuredMultiplyCompositions) {
  test_matrix m(15, 5);
  multiply_compositions(m.csr(), m.dense);
  multiply_compositions(m.csc(), m.dense);
  multiply_compositions(spblas::matrix_opt(m.csr()), m.dense);
  multiply_compositions(spblas::matrix_opt(m.csc()), m.dense);
}

namespace {

template <typename M>
constexpr bool solvable =
    requires(spblas::operation_state& s, M m, std::vector<T>& v) {
      spblas::triangular_solve(s, m, v, v);
    };

} // namespace

TEST(ViewAnalysis, WhatIsSolvable) {
  test_matrix m(4);
  auto a = m.csr();
  using A = decltype(a);
  static_assert(!solvable<A>);
  static_assert(!solvable<decltype(view::transpose(a))>);
  static_assert(
      solvable<decltype(view::transpose(view::triangle(a, uplo::lower)))>);
  static_assert(solvable<decltype(view::diagonal(a))>);
  // Not triangular.
  static_assert(!solvable<decltype(view::symmetric(a, uplo::lower))>);
  static_assert(!solvable<decltype(view::hermitian(a, uplo::lower))>);
  // Singular: rejected at compile time (a static_assert when called).
  static_assert(!spblas::__detail::triangular_structure<decltype(view::triangle(
                    a, uplo::lower, diag::implicit_zero))>);
  static_assert(!spblas::__detail::triangular_structure<decltype(view::diagonal(
                    a, diag::implicit_zero))>);
}
