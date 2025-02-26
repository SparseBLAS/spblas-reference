#ifndef MINIAPPS_CG_HH
#define MINIAPPS_CG_HH

#include <algorithm>
#include <iostream>
#include <tuple>
#include <utility>
#include <vector>

#include "krylov.hh"
#include "rl_blaspp.hh"
#include "util.hh"
#include <spblas/spblas.hpp>

namespace miniapps {

template <typename T> class CG : public Krylov<T> {
public:
  CG(T eps, int max_iters) : Krylov<T>{eps, max_iters} {}

  /**
   * CG or the conjugate gradient method is an iterative type Krylov subspace
   * method which is suitable for symmetric positive definite methods.
   *
   * Though this method performs very well for symmetric positive definite
   * matrices, it is in general not suitable for general matrices.
   *
   * We implement the CG variant from Section 2.3.1 of "Templates for the
   * Solution of Linear Systems: Building Blocks for Iterative Methods"
   *
   * @param[in] A  the matrix in CSR format
   * @param[in] b  the right hand side vector
   * @param[out] x  the solution vector
   */
  std::tuple<double, int> apply(const spblas::csr_view<T> a,
                                const std::vector<T> b,
                                std::vector<T> x) override;
};

// -----------------------------------------------------------------------------
template <typename T>
std::tuple<double, int> CG<T>::apply(const spblas::csr_view<T> a,
                                     const std::vector<T> b, std::vector<T> x) {
  int iters = 0;
  double error = 1.0;

  // r = b
  // rho = 0.0
  // prev_rho = 1.0
  // p = q = 0
  std::vector<T> r(b);
  std::vector<T> z(r);
  std::vector<T> p(b.size(), 0);
  std::vector<T> q(b.size(), 0);
  std::vector<T> tmp(b.size(), 0);

  double alpha, beta, rho;
  alpha = beta = rho = 0.0;
  double prev_rho = 1.0;

  // r = b-Ax
  spblas::multiply(a, x, tmp);
  blas::axpy(r.size(), -1.0, tmp.data(), 1, r.data(), 1);

  while (true) {
    // z = preconditioner*r
    z = r;
    // rho = dot(r, z)
    rho = blas::dot(r.size(), r.data(), 1, z.data(), 1);

    iters++;
    error = blas::nrm2(r.size(), r.data(), 1);
    if (iters >= this->max_iters || error < this->eps) {
      std::cout << "error = " << error << "\n";
      std::cout << "iters = " << iters << "\n";
      std::cout << "eps = " << this->eps << "\n";
      std::cout << "max_iters = " << this->max_iters << "\n";
      break;
    }

    // beta = rho / prev_rho;
    // p = z + beta * p
    beta = rho / prev_rho;
    tmp = p;
    blas::scal(tmp.size(), beta, tmp.data(), 1);
    blas::axpy(tmp.size(), 1.0, z.data(), 1, tmp.data(), 1);
    p = tmp;

    // q = A * p
    spblas::multiply(a, p, q);

    // alpha = rho / dot(p, q)
    alpha = rho / blas::dot(q.size(), p.data(), 1, q.data(), 1);

    // x = x + alpha * p
    // r = r - alpha * q
    blas::axpy(x.size(), alpha, p.data(), 1, x.data(), 1);
    blas::axpy(r.size(), -alpha, q.data(), 1, r.data(), 1);

    std::swap(rho, prev_rho);
  }
  return {error, iters};
}

} // namespace miniapps

#endif // MINIAPPS_CG_HH
