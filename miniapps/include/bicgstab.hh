#ifndef MINIAPPS_BICGSTAB_HH
#define MINIAPPS_BICGSTAB_HH

#include <algorithm>
#include <iostream>
#include <utility>
#include <vector>

#include "krylov.hh"
#include "rl_blaspp.hh"
#include "util.hh"
#include <spblas/spblas.hpp>

namespace miniapps {

template <typename T> class BiCGSTAB : public Krylov<T> {
public:
  BiCGSTAB(T eps, int max_iters) : Krylov<T>{eps, max_iters} {}

  /**
   * BiCGSTAB or the Bi-Conjugate Gradient-Stabilized is a Krylov subspace
   * solver.
   *
   * Being a generic solver, it is capable of solving general matrices,
   * including non-s.p.d matrices. Though, the memory and the computational
   * requirement of the BiCGSTAB solver are higher than of its s.p.d solver
   * counterpart, it has the capability to solve generic systems. It was
   * developed by stabilizing the BiCG method.
   *
   * We implement the BiCGSTAB variant from Section 2.3.8 of "Templates for t
   * e Solution of Linear Systems: Building Blocks for Iterative Methods"
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
std::tuple<double, int> BiCGSTAB<T>::apply(const spblas::csr_view<T> a,
                                           const std::vector<T> b,
                                           std::vector<T> x) {
  int iters = 0;
  double error = 1.0;

  // r = dense_b
  // prev_rho = rho = omega = alpha = beta = gamma = 1.0
  // rr = v = s = t = z = y = p = 0
  std::vector<T> r(b);
  std::vector<T> rr(b.size(), 0.0);
  std::vector<T> v(b.size(), 0.0);
  std::vector<T> s(b.size(), 0.0);
  std::vector<T> t(b.size(), 0.0);
  std::vector<T> z(b.size(), 0.0);
  std::vector<T> y(b.size(), 0.0);
  std::vector<T> p(b.size(), 0.0);
  std::vector<T> tmp(b.size(), 0.0);

  double prev_rho, rho, omega, alpha, beta, gamma;
  prev_rho = rho = omega = alpha = beta = gamma = 1.0;

  // r = b - Ax
  spblas::multiply(a, x, tmp);
  blas::axpy(r.size(), -1.0, tmp.data(), 1, r.data(), 1);
  // rr = r
  rr = r;
  while (true) {
    iters++;

    // rho = dot(rr, r)
    rho = blas::dot(r.size(), rr.data(), 1, r.data(), 1);

    error = blas::nrm2(r.size(), r.data(), 1);
    if (iters >= this->max_iters || error < this->eps) {
      std::cout << "error = " << error << "\n";
      std::cout << "iters = " << iters << "\n";
      std::cout << "eps = " << this->eps << "\n";
      std::cout << "max_iters = " << this->max_iters << "\n";
      break;
    }

    if (prev_rho * omega == 0.0) {
      p = r;
    } else {
      // beta = (rho / prev_rho) * (alpha / omega)
      beta = (rho / prev_rho) * (alpha / omega);
      // p = r + beta * (p - omega * v)
      tmp = p;
      blas::axpy(tmp.size(), -omega, v.data(), 1, tmp.data(), 1);
      blas::scal(tmp.size(), beta, tmp.data(), 1);
      blas::axpy(tmp.size(), 1.0, r.data(), 1, tmp.data(), 1);
      p = tmp;
    }

    // y = preconditioner * p
    y = p;
    // v = A * y
    spblas::multiply(a, y, v);
    // beta = dot(rr, v)
    beta = blas::dot(v.size(), rr.data(), 1, v.data(), 1);
    if (beta == 0.0) {
      s = r;
    } else {
      // alpha = rho / beta
      alpha = rho / beta;
      // s = r - alpha * v
      tmp = v;
      blas::scal(tmp.size(), -alpha, tmp.data(), 1);
      blas::axpy(tmp.size(), 1.0, r.data(), 1, tmp.data(), 1);
      s = tmp;
    }

    error = blas::nrm2(s.size(), s.data(), 1);
    if (iters >= this->max_iters || error < this->eps) {
      std::cout << "error = " << error << "\n";
      std::cout << "iters = " << iters << "\n";
      std::cout << "eps = " << this->eps << "\n";
      std::cout << "max_iters = " << this->max_iters << "\n";
      blas::axpy(x.size(), alpha, y.data(), 1, x.data(), 1);
      break;
    }

    // z = preconditioner * s
    z = s;
    // t = A * z
    spblas::multiply(a, z, t);
    // gamma = dot(t, s)
    gamma = blas::dot(s.size(), t.data(), 1, s.data(), 1);
    // beta = dot(t, t)
    beta = blas::dot(t.size(), t.data(), 1, t.data(), 1);
    // omega = gamma / beta
    if (beta == 0.0) {
      omega = 0.0;
    } else {
      omega = gamma / beta;
    }
    // x = x + alpha * y + omega * z
    tmp = z;
    blas::scal(tmp.size(), omega, tmp.data(), 1);
    blas::axpy(tmp.size(), alpha, y.data(), 1, tmp.data(), 1);
    blas::axpy(x.size(), 1.0, tmp.data(), 1, x.data(), 1);

    // r = s - omega * t
    tmp = t;
    blas::scal(tmp.size(), -omega, tmp.data(), 1);
    blas::axpy(tmp.size(), 1.0, s.data(), 1, tmp.data(), 1);
    r = tmp;

    std::swap(prev_rho, rho);
  }
  return {error, iters};
}

} // namespace miniapps

#endif // MINIAPPS_BICGSTAB_HH
