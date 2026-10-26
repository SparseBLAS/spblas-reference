#include <gtest/gtest.h>

#include <cstdlib>
#include <memory_resource>
#include <new>
#include <vector>

#include "util.hpp"
#include <spblas/spblas.hpp>

// Library-internal allocations must come from the operation_state's memory
// resource. This file replaces the global operator new to count calls made
// while an operation runs (other code is unaffected: counting is off
// otherwise), and gives each operation a resource that counts its
// allocations.

namespace {

thread_local bool counting_global_new = false;
thread_local std::size_t global_new_calls = 0;

// Number of global operator new calls made by f().
template <typename F>
std::size_t global_new_calls_during(F&& f) {
  global_new_calls = 0;
  counting_global_new = true;
  try {
    f();
  } catch (...) {
    counting_global_new = false;
    throw;
  }
  counting_global_new = false;
  return global_new_calls;
}

class counting_resource final : public std::pmr::memory_resource {
public:
  std::size_t allocations = 0;

private:
  void* do_allocate(std::size_t bytes, std::size_t alignment) override {
    allocations++;
    // Not a library allocation: don't count the upstream operator new.
    bool counting = counting_global_new;
    counting_global_new = false;
    void* p = std::pmr::new_delete_resource()->allocate(bytes, alignment);
    counting_global_new = counting;
    return p;
  }

  void do_deallocate(void* p, std::size_t bytes,
                     std::size_t alignment) override {
    std::pmr::new_delete_resource()->deallocate(p, bytes, alignment);
  }

  bool do_is_equal(const std::pmr::memory_resource& other) const noexcept {
    return this == &other;
  }
};

using T = double;
using I = spblas::index_t;

// One matrix in CSR and in CSC.
struct test_matrix {
  std::vector<T> values, csc_values;
  std::vector<I> rowptr, colind, colptr, rowind;
  spblas::index<I> shape;
  I nnz;

  test_matrix(I m, I n, I nnz_, std::size_t seed) : shape(m, n), nnz(nnz_) {
    auto [v, r, c, s, z] = spblas::generate_csr<T, I>(m, n, nnz_, seed);
    values = v;
    rowptr = r;
    colind = c;
    csc_values.resize(nnz);
    colptr.resize(n + 1);
    rowind.resize(nnz);
    spblas::csr_wrapper<T, I> t(csc_values, colptr, rowind, {n, m}, nnz);
    spblas::operation_state state;
    spblas::transpose(state, csr(), t);
  }

  spblas::csr_wrapper<T, I> csr() {
    return {values, rowptr, colind, shape, nnz};
  }

  spblas::csc_wrapper<T, I> csc() {
    return {csc_values, colptr, rowind, shape, nnz};
  }

  template <bool Csc>
  auto as() {
    if constexpr (Csc) {
      return csc();
    } else {
      return csr();
    }
  }
};

// Output arrays of a CSR / CSC result.
struct output {
  std::vector<T> values;
  std::vector<I> pointers, indices;

  template <typename C>
  void attach(C& c, std::size_t nnz, std::size_t outer, bool with_values) {
    values.assign(with_values ? nnz : 0, T(0));
    pointers.assign(outer + 1, I(0));
    indices.assign(nnz, I(0));
    c.update(values, pointers, indices);
  }

  void attach_values(auto& c) {
    values.assign(indices.size(), T(0));
    c.update(values, pointers, indices);
  }
};

// One state for all stages of an operation, as users run them; the stages
// are checked one by one.
template <bool CscA, bool CscB, bool CscC>
void spgemm_allocations_test() {
  test_matrix a_m(60, 40, 300, 0);
  test_matrix b_m(40, 50, 300, 1);
  auto a = a_m.as<CscA>();
  auto b = b_m.as<CscB>();
  using C = std::conditional_t<CscC, spblas::csc_wrapper<T, I>,
                               spblas::csr_wrapper<T, I>>;
  const std::size_t outer = CscC ? 50 : 60;

  {
    counting_resource resource;
    spblas::operation_state state(&resource);
    C c(60, 50);
    output out;
    EXPECT_EQ(global_new_calls_during(
                  [&] { spblas::multiply_compute(state, 2.0, a, b, c); }),
              0u)
        << "multiply_compute";
    out.attach(c, state.get_result_nnz(), outer, true);
    EXPECT_EQ(global_new_calls_during(
                  [&] { spblas::multiply_fill(state, 2.0, a, b, c); }),
              0u)
        << "multiply_fill";
    EXPECT_GT(resource.allocations, 0u);
  }
  {
    counting_resource resource;
    spblas::operation_state state(&resource);
    C c(60, 50);
    output out;
    EXPECT_EQ(global_new_calls_during([&] {
                spblas::multiply_compute_symbolic(state, 2.0, a, b, c);
              }),
              0u)
        << "multiply_compute_symbolic";
    out.attach(c, state.get_result_nnz(), outer, false);
    EXPECT_EQ(global_new_calls_during(
                  [&] { spblas::multiply_fill_symbolic(state, 2.0, a, b, c); }),
              0u)
        << "multiply_fill_symbolic";
    out.attach_values(c);
    EXPECT_EQ(global_new_calls_during(
                  [&] { spblas::multiply_numeric(state, 2.0, a, b, c); }),
              0u)
        << "multiply_numeric";
    EXPECT_GT(resource.allocations, 0u);
  }
}

template <bool Csc>
void spadd_allocations_test() {
  test_matrix a_m(60, 50, 300, 0);
  test_matrix b_m(60, 50, 300, 1);
  auto a = a_m.as<Csc>();
  auto b = b_m.as<Csc>();
  using C = std::conditional_t<Csc, spblas::csc_wrapper<T, I>,
                               spblas::csr_wrapper<T, I>>;
  const std::size_t outer = Csc ? 50 : 60;

  {
    counting_resource resource;
    spblas::operation_state state(&resource);
    C c(60, 50);
    output out;
    EXPECT_EQ(
        global_new_calls_during([&] { spblas::add_compute(state, a, b, c); }),
        0u)
        << "add_compute";
    out.attach(c, state.get_result_nnz(), outer, true);
    EXPECT_EQ(
        global_new_calls_during([&] { spblas::add_fill(state, a, b, c); }), 0u)
        << "add_fill";
  }
  {
    counting_resource resource;
    spblas::operation_state state(&resource);
    C c(60, 50);
    output out;
    EXPECT_EQ(global_new_calls_during(
                  [&] { spblas::add_compute_symbolic(state, a, b, c); }),
              0u)
        << "add_compute_symbolic";
    out.attach(c, state.get_result_nnz(), outer, false);
    EXPECT_EQ(global_new_calls_during(
                  [&] { spblas::add_fill_symbolic(state, a, b, c); }),
              0u)
        << "add_fill_symbolic";
    out.attach_values(c);
    EXPECT_EQ(
        global_new_calls_during([&] { spblas::add_numeric(state, a, b, c); }),
        0u)
        << "add_numeric";
    EXPECT_GT(resource.allocations, 0u);
  }
}

} // namespace

// Replaceable global allocation functions, counting while enabled: the
// plain and the aligned forms (the latter serve over-aligned types and
// std::pmr::new_delete_resource with a large alignment). The array, nothrow
// and sized forms default to these.
void* operator new(std::size_t bytes) {
  if (counting_global_new) {
    global_new_calls++;
  }
  if (void* p = std::malloc(bytes == 0 ? 1 : bytes)) {
    return p;
  }
  throw std::bad_alloc();
}

void* operator new(std::size_t bytes, std::align_val_t alignment) {
  if (counting_global_new) {
    global_new_calls++;
  }
  const std::size_t a = static_cast<std::size_t>(alignment);
  // aligned_alloc needs a size that is a multiple of the alignment.
  const std::size_t size = (bytes == 0 ? 1 : bytes + a - 1) / a * a;
  if (void* p = std::aligned_alloc(a, size)) {
    return p;
  }
  throw std::bad_alloc();
}

void operator delete(void* p) noexcept {
  std::free(p);
}

void operator delete(void* p, std::size_t) noexcept {
  std::free(p);
}

void operator delete(void* p, std::align_val_t) noexcept {
  std::free(p);
}

void operator delete(void* p, std::size_t, std::align_val_t) noexcept {
  std::free(p);
}

TEST(Allocator, SpGEMMUsesStateResource) {
  spgemm_allocations_test<false, false, false>();
  spgemm_allocations_test<false, false, true>();
  spgemm_allocations_test<true, true, true>();
  spgemm_allocations_test<false, true, false>();
  spgemm_allocations_test<false, true, true>();
  spgemm_allocations_test<true, false, false>();
  spgemm_allocations_test<true, false, true>();
}

TEST(Allocator, SpADDUsesStateResource) {
  spadd_allocations_test<false>();
  spadd_allocations_test<true>();
}

// Global operator new is counted while enabled, in its plain and aligned
// forms, so allocations that bypass the state's resource are detected.
TEST(Allocator, CountsGlobalNew) {
  EXPECT_EQ(global_new_calls_during([] { std::vector<int> v(10); }), 1u);

  struct alignas(64) over_aligned {
    char c[64];
  };
  EXPECT_EQ(global_new_calls_during([] { std::vector<over_aligned> v(2); }),
            1u);

  counting_resource resource;
  EXPECT_EQ(
      global_new_calls_during([&] { std::pmr::vector<int> v(10, &resource); }),
      0u);
  EXPECT_EQ(resource.allocations, 1u);

  // A std::pmr allocation through another resource than the state's is seen,
  // at the default and at a large alignment.
  auto* other = std::pmr::new_delete_resource();
  EXPECT_EQ(
      global_new_calls_during([&] { std::pmr::vector<int> v(10, other); }), 1u);
  EXPECT_EQ(global_new_calls_during([&] {
              void* p = other->allocate(256, 128);
              other->deallocate(p, 256, 128);
            }),
            1u);
}
