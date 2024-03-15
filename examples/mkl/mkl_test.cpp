#include <oneapi/mkl.hpp>
#include <spblas/spblas.hpp>

#include <fmt/ranges.h>

int main(int argc, char** argv) {
  using namespace spblas;

  using T = float;
  using I = std::int64_t;

  auto&& [values, rowptr, colind, shape, nnz] =
      generate_csr<T, I>(100, 100, 10);

  csr_view<T, I> a(values, rowptr, colind, shape, nnz);

  // Scale every value of `a` by 5 in place.
  scale(5.f, a);

  std::vector<T> b(100, 1);
  std::vector<T> c(100, 0);

  T alpha = 2.0f;
  // c = a * alpha * b
  multiply(a, scaled(alpha, b), c);

  sycl::queue q(sycl::cpu_selector_v);

  /*
   sycl::event gemv (sycl::queue                           &queue,
                     oneapi::mkl::transpose                transpose_val,
                     const fp                              alpha,
                     oneapi::mkl::sparse::matrix_handle_t  A_handle,
                     const fp                              *x,
                     const fp                              beta,
                     fp                                    *y,
                     const std::vector<sycl::event>        &dependencies = {});
*/
  std::vector<T> c_mkl(100, 0);

  oneapi::mkl::sparse::matrix_handle_t a_handle = nullptr;

  oneapi::mkl::sparse::init_matrix_handle(&a_handle);

  oneapi::mkl::sparse::set_csr_data(
      q, a_handle, a.shape()[0], a.shape()[1], oneapi::mkl::index_base::zero,
      a.rowptr().data(), a.colind().data(), a.values().data())
      .wait();

  oneapi::mkl::sparse::optimize_gemv(q, oneapi::mkl::transpose::nontrans,
                                     a_handle)
      .wait();

  oneapi::mkl::sparse::gemv(q, oneapi::mkl::transpose::nontrans, alpha,
                            a_handle, b.data(), 0.0f, c_mkl.data())
      .wait();

  oneapi::mkl::sparse::release_matrix_handle(q, &a_handle).wait();

  fmt::print("c (ref): {}\n", c);
  fmt::print("c (mkl): {}\n", c_mkl);

#ifdef ENABLE_ONEMKL
  fmt::print("Hello from oneMKL!\n");
#endif

  return 0;
}
