#ifndef MINIAPPS_KRYLOV_HH
#define MINIAPPS_KRYLOV_HH

#include <vector>

#include <spblas/spblas.hpp>

namespace miniapps {

/**
 * A very simple Krylov Solver interface for convenience.
 *
 * @tparam T  the value type
 */
template <typename T>
class Krylov {
 public:
  Krylov(T eps, int max_iters) : eps{eps}, max_iters{max_iters} {}

  virtual ~Krylov() {}

  virtual std::tuple<double, int> apply(const spblas::csr_view<T> a,
                                        const std::vector<T> b,
                                        std::vector<T> x) = 0;

  T get_eps() { return eps; }

  int get_max_iters() { return max_iters; }

  const T eps;
  const int max_iters;
};

}

#endif // MINIAPPS_KRYLOV_HH
