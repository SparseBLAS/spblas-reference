#include <gtest/gtest.h>

#include <complex>
#include <cstdint>
#include <memory_resource>
#include <vector>

#include "util.hpp"
#include <spblas/spblas.hpp>

namespace {

// Forwards to new/delete and counts calls.
class counting_resource : public std::pmr::memory_resource {
public:
  std::size_t allocations = 0;
  std::size_t deallocations = 0;

private:
  void* do_allocate(std::size_t bytes, std::size_t alignment) override {
    ++allocations;
    return std::pmr::new_delete_resource()->allocate(bytes, alignment);
  }

  void do_deallocate(void* p, std::size_t bytes,
                     std::size_t alignment) override {
    ++deallocations;
    std::pmr::new_delete_resource()->deallocate(p, bytes, alignment);
  }

  bool
  do_is_equal(const std::pmr::memory_resource& other) const noexcept override {
    return this == &other;
  }
};

// 3x3 matrix
//   [1 2 0]
//   [3 4 5]
//   [0 6 7]
struct small_csr {
  using T = float;
  using I = spblas::index_t;

  std::vector<T> values{1, 2, 3, 4, 5, 6, 7};
  std::vector<I> rowptr{0, 2, 5, 7};
  std::vector<I> colind{0, 1, 0, 1, 2, 1, 2};

  spblas::csr_wrapper<T, I> wrapper() {
    return spblas::csr_wrapper<T, I>(values, rowptr, colind, {3, 3}, 7);
  }
};

} // namespace

TEST(OperationState, DefaultResource) {
  spblas::operation_state state;

  static_assert(
      std::is_same_v<decltype(state.get_result_nnz()), spblas::offset_t>);
  EXPECT_EQ(state.get_result_nnz(), 0);
  EXPECT_EQ(state.get_allocator().resource(),
            spblas::__detail::default_memory_resource());
}

TEST(OperationState, UserResource) {
  counting_resource resource;

  spblas::operation_state state(&resource);
  EXPECT_EQ(state.get_allocator().resource(), &resource);

  auto allocator = state.get_allocator();
  void* p = allocator.allocate_bytes(100);
  allocator.deallocate_bytes(p, 100);
  EXPECT_EQ(resource.allocations, 1);
  EXPECT_EQ(resource.deallocations, 1);

  state.update_impl_(7);
  spblas::operation_state moved(std::move(state));
  EXPECT_EQ(moved.get_allocator().resource(), &resource);
  EXPECT_EQ(moved.get_result_nnz(), 7);

  spblas::operation_state from_allocator(
      std::pmr::polymorphic_allocator<>{&resource});
  EXPECT_EQ(from_allocator.get_allocator().resource(), &resource);

  spblas::operation_state from_null(nullptr);
  EXPECT_EQ(from_null.get_allocator().resource(),
            spblas::__detail::default_memory_resource());
}

TEST(OperationState, ResultNnzOverflowThrows) {
  using O = spblas::offset_t;
  spblas::operation_state state;

  if constexpr (std::is_unsigned_v<O>) {
    EXPECT_THROW(state.update_impl_(-1), std::overflow_error);
  }
  if constexpr (sizeof(O) < sizeof(std::int64_t)) {
    EXPECT_THROW(state.update_impl_(std::int64_t{1} << 40),
                 std::overflow_error);
  }
  EXPECT_EQ(state.get_result_nnz(), 0); // unchanged after a failed update

  state.update_impl_(std::int64_t{42});
  EXPECT_EQ(state.get_result_nnz(), 42);
}

TEST(MatrixOpt, SharedStateFromUserResource) {
  small_csr m;
  counting_resource resource;

  {
    spblas::matrix_opt a_opt(m.wrapper(), &resource);
    EXPECT_GE(resource.allocations, 1);
    EXPECT_EQ(a_opt.get_allocator().resource(), &resource);

    auto copy = a_opt;
    EXPECT_EQ(copy.state_.get(), a_opt.state_.get());

    EXPECT_EQ(spblas::__backend::shape(a_opt),
              (spblas::index<spblas::index_t>(3, 3)));
    EXPECT_EQ(spblas::__backend::size(a_opt), 7);
  }
  EXPECT_EQ(resource.allocations, resource.deallocations);

  spblas::matrix_opt default_opt(m.wrapper());
  EXPECT_EQ(default_opt.get_allocator().resource(),
            spblas::__detail::default_memory_resource());
}

TEST(CsrWrapper, ShapeOnlyAndUpdate) {
  using T = float;
  using I = spblas::index_t;

  static_assert(
      std::is_same_v<spblas::csr_view<T, I>, spblas::csr_wrapper<T, I>>);
  static_assert(spblas::__detail::is_csr_wrapper_v<spblas::csr_wrapper<T, I>>);
  static_assert(spblas::__detail::is_csr_view_v<spblas::csr_wrapper<T, I>>);
  static_assert(spblas::matrix<spblas::csr_wrapper<T, I>>);

  spblas::csr_wrapper<T, I> c(4, 5);
  EXPECT_EQ(c.shape(), (spblas::index<I>(4, 5)));
  EXPECT_EQ(c.size(), 0);
  EXPECT_TRUE(c.rowptr().empty());

  std::vector<T> values(6);
  std::vector<I> rowptr(5);
  std::vector<I> colind(6);
  c.update(values, rowptr, colind);
  EXPECT_EQ(c.size(), 6);

  // Pattern first, values later (3-stage SpGEMM): nnz follows colind.
  spblas::csr_wrapper<T, I> pattern({4, 5});
  pattern.update({}, rowptr, colind);
  EXPECT_EQ(pattern.size(), 6);
}

TEST(CscWrapper, ShapeOnlyAndUpdate) {
  using T = float;
  using I = spblas::index_t;

  static_assert(
      std::is_same_v<spblas::csc_view<T, I>, spblas::csc_wrapper<T, I>>);
  static_assert(spblas::__detail::is_csc_wrapper_v<spblas::csc_wrapper<T, I>>);
  static_assert(spblas::__detail::is_csc_view_v<spblas::csc_wrapper<T, I>>);

  spblas::csc_wrapper<T, I> c(4, 5);
  EXPECT_EQ(c.shape(), (spblas::index<I>(4, 5)));
  EXPECT_EQ(c.size(), 0);

  std::vector<T> values(3);
  std::vector<I> colptr(6);
  std::vector<I> rowind(3);
  c.update(values, colptr, rowind);
  EXPECT_EQ(c.size(), 3);
}

TEST(View, Triangle) {
  small_csr m;
  auto a = m.wrapper();

  auto lower = spblas::view::triangle(a, spblas::view::uplo::lower);
  static_assert(spblas::__detail::is_triangle_view_v<decltype(lower)>);
  static_assert(spblas::matrix<decltype(lower)>);
  // Must not be unwrapped transparently by backends that match CSR bases.
  static_assert(!spblas::__detail::has_csr_base<decltype(lower)>);

  EXPECT_EQ(lower.uplo(), spblas::view::uplo::lower);
  EXPECT_EQ(lower.diag(), spblas::view::diag::explicit_diagonal);
  EXPECT_EQ(spblas::__backend::shape(lower),
            (spblas::index<spblas::index_t>(3, 3)));
  EXPECT_EQ(spblas::__backend::size(lower), 5);

  auto upper = spblas::view::triangle(a, spblas::view::uplo::upper);
  EXPECT_EQ(upper.size(), 5);

  // An implied unit diagonal counts (2 strictly lower entries + 3 diagonal
  // positions); an implied zero diagonal does not (below).
  auto unit_lower = spblas::view::triangle(a, spblas::view::uplo::lower,
                                           spblas::view::diag::implicit_unit);
  EXPECT_EQ(unit_lower.diag(), spblas::view::diag::implicit_unit);
  EXPECT_EQ(unit_lower.size(), 5);

  // Through a matrix_opt and through a CSC (column-iterable) matrix.
  spblas::matrix_opt a_opt(a);
  EXPECT_EQ(spblas::view::triangle(a_opt, spblas::view::uplo::upper).size(), 5);

  auto a_t = spblas::view::transpose(a); // CSC of the transpose
  EXPECT_EQ(spblas::view::triangle(a_t, spblas::view::uplo::upper,
                                   spblas::view::diag::implicit_zero)
                .size(),
            2);
}

TEST(View, TransposeAndConjugate) {
  small_csr m;
  auto a = m.wrapper();

  auto a_t = spblas::view::transpose(a);
  static_assert(spblas::__detail::is_csc_wrapper_v<decltype(a_t)>);
  EXPECT_EQ(a_t.shape(), (spblas::index<spblas::index_t>(3, 3)));
  EXPECT_EQ(a_t.size(), 7);

  // Real value types: conjugate is the identity.
  static_assert(
      std::is_same_v<decltype(spblas::view::conjugate(a)), decltype(a)>);

  std::vector<std::complex<float>> values(m.values.begin(), m.values.end());
  spblas::csr_wrapper<std::complex<float>, spblas::index_t> c(
      values, m.rowptr, m.colind, {3, 3}, 7);
  auto c_conj = spblas::view::conjugate(c);
  EXPECT_TRUE(spblas::__detail::is_conjugated(c_conj));

  // conj(conj(A)) is A, for named and temporary views, matrices and vectors.
  auto c_conj_conj = spblas::view::conjugate(c_conj);
  static_assert(std::is_same_v<decltype(c_conj_conj), decltype(c)>);
  static_assert(std::is_same_v<decltype(spblas::view::conjugate(
                                   spblas::view::conjugate(c))),
                               decltype(c)>);
  EXPECT_EQ(c_conj_conj.values().data(), values.data());
  std::vector<std::complex<float>> v(3);
  auto v_conj_conj = spblas::conjugated(spblas::conjugated(v));
  EXPECT_FALSE(spblas::__detail::is_conjugated(v_conj_conj));
}

// view::conjugate of a real vector is a non-owning identity view: it refers
// to the vector instead of copying it.
TEST(View, ConjugateOfRealVectorRefersToIt) {
  std::vector<double> x{1, 2, 3};
  auto v = spblas::view::conjugate(x);
  static_assert(spblas::__detail::__ranges::view<decltype(v)>);
  EXPECT_EQ(v.data(), x.data());

  x[1] = 20;
  EXPECT_EQ(v[1], 20);

  // Complex vectors were already views.
  std::vector<std::complex<double>> z{{1, 2}};
  auto w = spblas::view::conjugate(z);
  z[0] = {3, 4};
  EXPECT_EQ(w[0], std::complex<double>(3, -4));
}

TEST(View, TransposeOfMatrixOpt) {
  using T = float;
  using I = spblas::index_t;

  auto [values, rowptr, colind, shape, nnz] =
      spblas::generate_csr<T, I>(4, 6, 10);
  spblas::csr_wrapper<T, I> a(values, rowptr, colind, shape, nnz);
  spblas::matrix_opt a_opt(a);

  auto t = spblas::view::transpose(a_opt);
  static_assert(spblas::__detail::is_transposed_view_v<decltype(t)>);
  static_assert(spblas::matrix<decltype(t)>);
  // Must not be unwrapped transparently by backends that match CSR bases.
  static_assert(!spblas::__detail::has_csr_base<decltype(t)>);
  static_assert(!spblas::__detail::has_csc_base<decltype(t)>);

  // The optimization data stays attached.
  EXPECT_EQ(t.matrix().state_.get(), a_opt.state_.get());
  EXPECT_EQ(spblas::__backend::shape(t), (spblas::index<I>(6, 4)));
  EXPECT_EQ(spblas::__backend::size(t), nnz);
  EXPECT_EQ(util::to_dense(t), util::to_dense(spblas::view::transpose(a)));

  // Transposing twice gives back the matrix_opt itself.
  auto tt = spblas::view::transpose(t);
  static_assert(std::is_same_v<decltype(tt), decltype(a_opt)>);
  EXPECT_EQ(tt.state_.get(), a_opt.state_.get());
}

TEST(View, TransposeOfDense) {
  using T = double;
  using I = spblas::index_t;

  std::vector<T> x_values{1, 2, 3, 4, 5, 6};
  spblas::mdspan_row_major<T, I> x(x_values.data(), 2, 3);
  auto x_t = spblas::view::transpose(x);

  EXPECT_EQ(spblas::__backend::shape(x_t), (spblas::index<I>(3, 2)));
  for (I i = 0; i < 2; i++) {
    for (I j = 0; j < 3; j++) {
      EXPECT_EQ(spblas::__backend::lookup(x_t, j, i), x_values[i * 3 + j]);
    }
  }
}
