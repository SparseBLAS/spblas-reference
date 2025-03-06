#include "miniapps.hh"
#include <spblas/spblas.hpp>

#include <algorithm>
#include <exception>
#include <vector>

#include <gtest/gtest.h>

template <typename T>
class TestCG : public testing::Test {
public:
  using ValueType = T;

  auto generate_spd_problem(std::size_t m, std::size_t n, std::size_t nnz_input,
                            std::size_t seed = 0) {
    auto&& [values_orig, rowind_orig, colind_orig, shape, nnz_orig] =
        spblas::generate_coo<T, spblas::index_t>(m, n, nnz_input, seed);
    miniapps::matrix_data<T> data(rowind_orig, colind_orig, values_orig, shape);
    data.sort_row_major();
    data.make_symmetric();
    data.make_diag_dominant();
    auto&& [values, rowind, colind, nnz] = data.convert_to_coo();
    auto rowptr = miniapps::convert_rowind_to_rowptr(colind, nnz, shape);
    return std::tuple(values, rowptr, colind, shape, nnz);
  }
};

using CGTestTypes = ::testing::Types<double, float>;
TYPED_TEST_SUITE(TestCG, CGTestTypes);

TYPED_TEST(TestCG, ConvergesForSmallSystem) {
  using T = TypeParam;
  constexpr double tol = std::is_same<T, double>::value ? 1e-14 : 1e-7;
  constexpr int max_iters = 100;
  std::vector<T> b(10, 1.0);
  std::vector<T> x(10, 0.0);

  auto&& [values, rowptr, colind, shape, nnz] =
      this->generate_spd_problem(10, 10, 42, 75);
  spblas::csr_view<T> a(values, rowptr, colind, shape, nnz);

  miniapps::CG<T> CG(tol, max_iters);
  auto [error, iters] = CG.apply(a, b, x);

  ASSERT_LE(error, tol);
  ASSERT_LE(iters, max_iters);
}

TYPED_TEST(TestCG, ConvergesForLargeSystem) {
  using T = TypeParam;
  constexpr double tol = std::is_same<T, double>::value ? 1e-14 : 1e-7;
  constexpr int max_iters = 100;
  std::vector<T> b(1000, 1.0);
  std::vector<T> x(1000, 0.0);

  auto&& [values, rowptr, colind, shape, nnz] =
      this->generate_spd_problem(1000, 1000, 12345, 75);
  spblas::csr_view<T> a(values, rowptr, colind, shape, nnz);

  miniapps::CG<T> CG(tol, max_iters);
  auto [error, iters] = CG.apply(a, b, x);

  ASSERT_LE(error, tol);
  ASSERT_LE(iters, max_iters);
}
