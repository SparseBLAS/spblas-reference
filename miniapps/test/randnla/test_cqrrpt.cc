#include "miniapps.hh"
#include "rl_blaspp.hh"
#include "rl_lapackpp.hh"
#include <spblas/spblas.hpp>

#include <fstream>
#include <gtest/gtest.h>

class TestCQRRPT : public ::testing::Test {
protected:
  virtual void SetUp(){};

  virtual void TearDown(){};

  template <typename T>
  struct CQRRPTTestData {
    int64_t row;
    int64_t col;
    int64_t rank;
    std::vector<T> A;
    std::vector<T> R;
    std::vector<int64_t> J;
    std::vector<T> A_cpy1;
    std::vector<T> A_cpy2;
    std::vector<T> I_ref;

    CQRRPTTestData(int64_t m, int64_t n, int64_t k)
        : A(m * n, 0.0), R(n * n, 0.0), J(n, 0), A_cpy1(m * n, 0.0),
          A_cpy2(m * n, 0.0), I_ref(k * k, 0.0) {
      row = m;
      col = n;
      rank = k;
    }
  };

  template <typename T>
  static void norm_and_copy_computational_helper(T& norm_A,
                                                 CQRRPTTestData<T>& all_data) {
    auto m = all_data.row;
    auto n = all_data.col;

    lapack::lacpy(MatrixType::General, m, n, all_data.A.data(), m,
                  all_data.A_cpy1.data(), m);
    lapack::lacpy(MatrixType::General, m, n, all_data.A.data(), m,
                  all_data.A_cpy2.data(), m);
    norm_A = lapack::lange(Norm::Fro, m, n, all_data.A.data(), m);
  }

  template <typename T>
  static void error_check(T& norm_A, CQRRPTTestData<T>& all_data) {

    auto m = all_data.row;
    auto n = all_data.col;
    auto k = all_data.rank;

    all_data.I_ref.resize(k * k, 0.0);
    for (int i = 0; i < k; ++i)
      all_data.I_ref[(k + 1) * i] = 1.0;

    T* A_dat = all_data.A_cpy1.data();
    T const* A_cpy_dat = all_data.A_cpy2.data();
    T const* Q_dat = all_data.A.data();
    T const* R_dat = all_data.R.data();
    T* I_ref_dat = all_data.I_ref.data();

    // Check orthogonality of Q
    // Q' * Q  - I = 0
    blas::syrk(Layout::ColMajor, Uplo::Upper, Op::Trans, k, m, 1.0, Q_dat, m,
               -1.0, I_ref_dat, k);
    T norm_QTQ = lapack::lansy(lapack::Norm::Fro, Uplo::Upper, k, I_ref_dat, k);

    // A - QR
    blas::gemm(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, n, k, 1.0, Q_dat,
               m, R_dat, n, -1.0, A_dat, m);

    // Implementing max col norm metric
    T max_col_norm = 0.0;
    T col_norm = 0.0;
    int max_idx = 0;
    for (int i = 0; i < n; ++i) {
      col_norm = blas::nrm2(m, &A_dat[m * i], 1);
      if (max_col_norm < col_norm) {
        max_col_norm = col_norm;
        max_idx = i;
      }
    }
    T col_norm_A = blas::nrm2(n, &A_cpy_dat[m * max_idx], 1);
    T norm_AQR = lapack::lange(Norm::Fro, m, n, A_dat, m);

    printf("REL NORM OF AP - QR:    %15e\n", norm_AQR / norm_A);
    printf("MAX COL NORM METRIC:    %15e\n", max_col_norm / col_norm_A);
    printf("FRO NORM OF (Q'Q - I):  %2e\n\n", norm_QTQ);

    T atol = std::pow(std::numeric_limits<T>::epsilon(), 0.75);
    ASSERT_NEAR(norm_AQR / norm_A, 0.0, atol);
    ASSERT_NEAR(max_col_norm / col_norm_A, 0.0, atol);
    ASSERT_NEAR(norm_QTQ, 0.0, atol);
  }

  /// General test for CQRRPT:
  /// Computes QR factorzation, and computes A[:, J] - QR.
  template <typename T, typename alg_type>
  static void test_CQRRPT_general(T d_factor, T norm_A,
                                  CQRRPTTestData<T>& all_data,
                                  alg_type& CQRRPT) {

    auto m = all_data.row;
    auto n = all_data.col;

    CQRRPT.call(m, n, all_data.A.data(), m, all_data.R.data(), n,
                all_data.J.data(), d_factor);
    all_data.rank = CQRRPT.rank;

    printf("RANK AS RETURNED BY CQRRPT %ld\n", all_data.rank);

    miniapps::util::col_swap(m, n, n, all_data.A_cpy1.data(), m, all_data.J);
    miniapps::util::col_swap(m, n, n, all_data.A_cpy2.data(), m, all_data.J);

    error_check(norm_A, all_data);
  }
};

TEST_F(TestCQRRPT, CQRRPT_full_rank_no_hqrrp) {
  int64_t m = 50;
  int64_t n = 20;
  int64_t k = 20;
  double d_factor = 1.25;
  double norm_A = 0;
  int64_t nnz = 2;
  double tol = std::pow(std::numeric_limits<double>::epsilon(), 0.85);

  CQRRPTTestData<double> all_data(m, n, k);
  miniapps::CQRRPT<double> CQRRPT(tol, nnz);
  // Generate dense matrix
  auto [buf, a_shape] = spblas::generate_dense<double>(m, n);

  lapack::lacpy(MatrixType::General, m, n, buf.data(), m, all_data.A.data(), m);

  norm_and_copy_computational_helper<double>(norm_A, all_data);
  test_CQRRPT_general<double, miniapps::CQRRPT<double>>(d_factor, norm_A,
                                                        all_data, CQRRPT);
}
