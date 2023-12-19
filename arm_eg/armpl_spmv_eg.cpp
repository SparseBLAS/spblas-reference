// armclang++ -DINTEGER64 -std=c++23 -I include -I _deps/mdspan-src/include -I _deps/range-v3-src/include -I $ARMPL_DIR/include arm_eg/armpl_spmv_eg.cpp $ARMPL_DIR/lib/libarmpl_ilp64.a
#include <iostream>
#include <vector>
#include <spblas/spblas.hpp>

#define NNZ 12
#define M 5
#define N 5

int main(int argc, char** argv) {

  using namespace spblas;

  int nnz = 12;
  int m = 5;
  int n = 5;
  std::vector<float> values = {1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.};
  std::vector<size_t> rowptr = {0, 2, 4, 7, 9, 12}; // XXX size_t only option at present - needs our ilp64 lib
  std::vector<size_t> colind = {0, 2, 1, 3, 1, 2, 3, 2, 3, 2, 3, 4};
  index<size_t> shape = {m, n};

  csr_view<float> a(values.data(), rowptr.data(), colind.data(), shape, nnz);

  std::vector<float> b(n, 1);
  std::vector<float> c(m, 0);
  float alpha = 2.0f;

  // XXX hints here?
  auto info = multiply_inspect(scaled(alpha, a), b, c);

  //multiply(a, scaled(alpha, b), c);                                        // a is csr_view type - unoptimized
  //multiply(a, b, c);                                                       // ditto

  //multiply(scaled(alpha, a),  b, c);                                       // OK! We can return our own view for alpha*a which wraps armpl_spmat_t and alpha; no info involved though
  multiply(bind_info{scaled(alpha, a), std::move(info)}, b, c);            // repeating `scaled` call here seems like a problem!? XXX // std::move(info) if we don't want copies - potentially heavyweight!
  //multiply(bind_info{scaled(alpha, a), info}, b, c);                       // same, but without std::move (info is not heavyweight really in this particular impl...)

  for (auto v : c) {
    std::cout << v << std::endl;
  }

  return 0;
}
