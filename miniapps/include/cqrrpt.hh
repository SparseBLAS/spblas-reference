#ifndef miniapps_cqrrpt_h
#define miniapps_cqrrpt_h

#include <cstdint>
#include <vector>
#include <chrono>
#include <numeric>

#include "util.hh"
#include "rl_blaspp.hh"
#include "rl_lapackpp.hh"
#include <spblas/spblas.hpp>

namespace miniapps {

template <typename T>
class CQRRPTalg {
    public:

        virtual ~CQRRPTalg() {}

        virtual int call(
            int64_t m,
            int64_t n,
            T* A,
            int64_t lda,
            T* R,
            int64_t ldr,
            int64_t* J,
            T d_factor
        ) = 0;
};

template <typename T>
class CQRRPT : public CQRRPTalg<T> {
    public:
        CQRRPT(T ep, int64_t nz) {
            eps = ep;
            nnz = nz;
        }

        /// Computes a QR factorization with column pivots of the form:
        ///     A[:, J] = QR,
        /// where Q and R are of size m-by-k and k-by-n, with rank(A) = k.
        /// Detailed description of this algorithm may be found in Section 5.1.2.
        /// of "the RandLAPACK book". 
        ///
        /// @param[in] m
        ///     The number of rows in the matrix A.
        ///
        /// @param[in] n
        ///     The number of columns in the matrix A.
        ///
        /// @param[in] A
        ///     The m-by-n matrix A, stored in a column-major format.
        ///
        /// @param[in] d
        ///     Embedding dimension of a sketch, m >= d >= n.
        ///
        /// @param[in] R
        ///     Represents the upper-triangular R factor of QR factorization.
        ///     On entry, is empty and may not have any space allocated for it.
        ///
        /// @param[out] A
        ///     Overwritten by an m-by-k orthogonal Q factor.
        ///     Matrix is stored explicitly.
        ///
        /// @param[out] R
        ///     Stores k-by-n matrix with upper-triangular R factor.
        ///     Zero entries are not compressed.
        ///
        /// @param[out] J
        ///     Stores k integer type pivot index extries.
        ///
        /// @return = 0: successful exit
        ///
        /// @return = 1: cholesky factorization failed
        ///

        int call(
            int64_t m,
            int64_t n,
            T* A,
            int64_t lda,
            T* R,
            int64_t ldr,
            int64_t* J,
            T d_factor
        ) override;

    public:
        T eps;
        int64_t nnz;
        int64_t rank;
};

// -----------------------------------------------------------------------------
template <typename T>
int CQRRPT<T>::call(
    int64_t m,
    int64_t n,
    T* A,
    int64_t lda,
    T* R,
    int64_t ldr,
    int64_t* J,
    T d_factor
){
    int i;
    int64_t k = n;
    int64_t d = d_factor * n;
    // A constant for initial rank estimation.
    T eps_initial_rank_estimation = 2 * std::pow(std::numeric_limits<T>::epsilon(), 0.95);
    // Variables for a posteriori rank estimation.
    int64_t new_rank;
    T running_max, running_min, curr_entry;

    T* A_hat = ( T * ) calloc( d * n, sizeof( T ) );
    T* tau   = ( T * ) calloc( n, sizeof( T ) );
    // Buffer for column pivoting.
    std::vector<int64_t> J_buf(n, 0);
    
    /* RandBLAS style
    /// Generating a SASO
    RandBLAS::SparseDist DS = {.n_rows = d, .n_cols = m, .vec_nnz = this->nnz};
    RandBLAS::SparseSkOp<T, RNG> S(DS, state);
    state = RandBLAS::fill_sparse(S);

    /// Applying a SASO
    RandBLAS::sketch_general(
        Layout::ColMajor, Op::NoTrans, Op::NoTrans,
        d, n, m, 1.0, S, 0, 0, A, lda, 0.0, A_hat, d
    );
    */

    /// SparseBLAS style
    /// Our sketching operators must be in COO format.
    //auto&& [values, rowptr, colind, shape, _] = spblas::generate_coo<T>(d, m, nnz);
    /// TODO: add COO view
    //spblas::coo_view<T> s(values, rowptr, colind, shape, nnz);
    
    /// Perform dense sketching for the sake of tests passing.
    /// This is not the intended behavior for this algorithm, as dense sketching 
    /// tanks the performance of CQRRPT.
    auto [buf, a_shape] = spblas::generate_gaussian<double>(d, m);
    spblas::__mdspan::mdspan s(buf.data(), d, m);

    spblas::__mdspan::mdspan a(A, m, n);
    spblas::__mdspan::mdspan ahat(A_hat, d, n);

    spblas::multiply(s, a, ahat);
    A_hat = ahat.data_handle();

    /// Performing QRCP on a sketch
    lapack::geqp3(d, n, A_hat, d, J, tau);

    /// Using naive rank estimation to ensure that R used for preconditioning is invertible.
    /// The actual rank estimate k will be computed a posteriori. 
    /// Using R[i,i] to approximate the i-th singular value of A_hat. 
    /// Truncate at the largest i where R[i,i] / R[0,0] >= eps.
    for(i = 0; i < n; ++i) {
        if(std::abs(A_hat[i * d + i]) / std::abs(A_hat[0]) < eps_initial_rank_estimation) {
            k = i;
            break;
        }
    }
    this->rank = k;

    // Allocating space for a preconditioner buffer.
    T* R_sp  = ( T * ) calloc( k * k, sizeof( T ) );
    /// Extracting a k by k upper-triangular R.
    lapack::lacpy(MatrixType::Upper, k, k, A_hat, d, R_sp, k);
    /// Extracting a k by n R representation (k by k upper-triangular, rest - general)
    lapack::lacpy(MatrixType::Upper, k, k, A_hat, d, R, ldr);
    lapack::lacpy(MatrixType::General, k, n - k, &A_hat[d * k], d, &R[n * k], ldr);

    // Swap k columns of A with pivots from J
    blas::copy(n, J, 1, J_buf.data(), 1);
    util::col_swap(m, n, k, A, lda, J_buf);

    // A_pre * R_sp = AP
    blas::trsm(Layout::ColMajor, Side::Right, Uplo::Upper, Op::NoTrans, Diag::NonUnit, m, k, 1.0, R_sp, k, A, lda);

    // Do Cholesky QR
    blas::syrk(Layout::ColMajor, Uplo::Upper, Op::Trans, k, m, 1.0, A, lda, 0.0, R_sp, k);
    lapack::potrf(Uplo::Upper, k, R_sp, k);

    // Re-estimate rank after we have the R-factor form Cholesky QR.
    // The strategy here is the same as in naive rank estimation.
    // This also automatically takes care of any potentical failures in Cholesky factorization.
    // Note that the diagonal of R_sp may not be sorted, so we need to keep the running max/min
    // We expect the loss in the orthogonality of Q to be approximately equal to u * cond(R_sp)^2, where u is the unit roundoff for the numerical type T.
    new_rank = k;
    running_max = R_sp[0];
    running_min = R_sp[0];
    
    for(i = 0; i < k; ++i) {
        curr_entry = std::abs(R_sp[i * k + i]);
        running_max = std::max(running_max, curr_entry);
        running_min = std::min(running_min, curr_entry);
        if(running_max / running_min >= std::sqrt(this->eps / std::numeric_limits<T>::epsilon())) {
            new_rank = i - 1;
            break;
        }
    }

    blas::trsm(Layout::ColMajor, Side::Right, Uplo::Upper, Op::NoTrans, Diag::NonUnit, m, new_rank, 1.0, R_sp, k, A, lda);
    // Get the final R-factor.
    blas::trmm(Layout::ColMajor, Side::Left, Uplo::Upper, Op::NoTrans, Diag::NonUnit, new_rank, n, 1.0, R_sp, k, R, ldr);

    // Set the rank parameter to the value comuted a posteriori.
    this->rank = k;

    free(A_hat);
    free(R_sp);
    free(tau);

    return 0;
}
} // end namespace miniapps
#endif
