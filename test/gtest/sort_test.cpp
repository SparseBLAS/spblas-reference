#include <gtest/gtest.h>

#include <complex>
#include <cstdint>
#include <memory_resource>
#include <new>
#include <vector>

#include "util.hpp"
#include <spblas/spblas.hpp>

namespace {

// Forwards to new/delete, but throws std::bad_alloc once it has served
// `allowed` allocations.
class failing_resource : public std::pmr::memory_resource {
public:
  explicit failing_resource(std::size_t allowed) : allowed_(allowed) {}

private:
  void* do_allocate(std::size_t bytes, std::size_t alignment) override {
    if (allowed_ == 0) {
      throw std::bad_alloc();
    }
    --allowed_;
    return std::pmr::new_delete_resource()->allocate(bytes, alignment);
  }

  void do_deallocate(void* p, std::size_t bytes,
                     std::size_t alignment) override {
    std::pmr::new_delete_resource()->deallocate(p, bytes, alignment);
  }

  bool
  do_is_equal(const std::pmr::memory_resource& other) const noexcept override {
    return this == &other;
  }

  std::size_t allowed_;
};

template <typename T, typename I>
void sort_csr_test() {
  for (auto&& [m, n, nnz] : util::dims) {
    auto [values, rowptr, colind, shape, _] =
        spblas::generate_csr<T, I>(m, n, nnz);
    util::shuffle_slices(rowptr, colind, values);
    spblas::csr_wrapper<T, I> a(values, rowptr, colind, shape, nnz);
    auto before = util::to_dense(a);

    spblas::operation_state state;
    spblas::sort_matrix(state, a);

    EXPECT_TRUE(util::sorted_slices(rowptr, colind));
    EXPECT_EQ(util::to_dense(a), before);
  }
}

} // namespace

TEST(SortMatrix, Csr) {
  sort_csr_test<float, spblas::index_t>();
  sort_csr_test<std::complex<double>, spblas::index_t>();
  sort_csr_test<double, std::int64_t>();
}

TEST(SortMatrix, CscAndPolicy) {
  using T = double;
  using I = spblas::index_t;

  for (auto&& [m, n, nnz] : util::dims) {
    auto [values, colptr, rowind, shape, _] =
        spblas::generate_csc<T, I>(m, n, nnz);
    util::shuffle_slices(colptr, rowind, values);
    spblas::csc_wrapper<T, I> a(values, colptr, rowind, shape, nnz);
    auto before = util::to_dense(a);

    spblas::operation_state state;
    spblas::sort_matrix(util::test_policy{}, state, a);

    EXPECT_TRUE(util::sorted_slices(colptr, rowind));
    EXPECT_EQ(util::to_dense(a), before);
  }
}

// If an allocation throws, the matrix the arrays represent is unchanged (each
// row is sorted or untouched), and sorting again finishes the job. Before,
// a row's indices could be written back before its value buffer was
// allocated: indices {1, 0} with values {10, 20} became {0, 1} with
// {10, 20}, and a retry skipped the now-sorted row.
TEST(SortMatrix, AllocationFailureKeepsTheMatrix) {
  using T = double;
  using I = spblas::index_t;

  auto check = [](std::vector<T> values, std::vector<I> rowptr,
                  std::vector<I> colind, spblas::index<I> shape) {
    const I nnz = I(colind.size());
    spblas::csr_wrapper<T, I> a(values, rowptr, colind, shape, nnz);
    const auto before = util::to_dense(a);

    bool sorted = false;
    for (std::size_t allowed = 0; !sorted; allowed++) {
      failing_resource resource(allowed);
      spblas::operation_state state(&resource);
      try {
        spblas::sort_matrix(state, a);
        sorted = true;
      } catch (const std::bad_alloc&) {
        EXPECT_EQ(util::to_dense(a), before)
            << "after a failure with " << allowed << " allocations allowed";
        spblas::operation_state retry;
        auto values_copy = values;
        auto colind_copy = colind;
        spblas::csr_wrapper<T, I> b(values_copy, rowptr, colind_copy, shape,
                                    nnz);
        spblas::sort_matrix(retry, b);
        EXPECT_TRUE(util::sorted_slices(rowptr, colind_copy));
        EXPECT_EQ(util::to_dense(b), before);
      }
    }
    EXPECT_TRUE(util::sorted_slices(rowptr, colind));
    EXPECT_EQ(util::to_dense(a), before);
  };

  // The reported case: one row, indices {1, 0}, values {10, 20}.
  check({10, 20}, {0, 2}, {1, 0}, {1, 2});

  for (auto&& [m, n, nnz] : util::dims) {
    auto [values, rowptr, colind, shape, _] =
        spblas::generate_csr<T, I>(m, n, nnz);
    util::shuffle_slices(rowptr, colind, values);
    check(values, rowptr, colind, shape);
  }
}

TEST(SortMatrix, MatrixOptStaysUsable) {
  using T = double;
  using I = spblas::index_t;

  auto [m, n, nnz] = util::dims[1];
  auto [values, rowptr, colind, shape, _] =
      spblas::generate_csr<T, I>(m, n, nnz);
  util::shuffle_slices(rowptr, colind, values);
  spblas::csr_wrapper<T, I> a(values, rowptr, colind, shape, nnz);
  spblas::matrix_opt a_opt(a);

  auto x = util::make_values<T>(n);
  std::vector<T> y(m), y_sorted(m);
  spblas::operation_state state;
  spblas::multiply_inspect(state, T(1), a_opt, x, y);
  spblas::multiply(state, T(1), a_opt, x, y);

  spblas::sort_matrix(state, a_opt);
  EXPECT_TRUE(util::sorted_slices(rowptr, colind));
  spblas::sort_matrix(state, a_opt); // already sorted
  EXPECT_TRUE(util::sorted_slices(rowptr, colind));

  spblas::multiply(state, T(1), a_opt, x, y_sorted);
  for (std::size_t i = 0; i < m; i++) {
    EXPECT_NEAR(y[i], y_sorted[i], 1e-12 * (1 + std::abs(y[i])));
  }
}

TEST(SortMatrix, PatternOnly) {
  using T = float;
  using I = spblas::index_t;

  std::vector<I> rowptr{0, 3, 3, 5};
  std::vector<I> colind{4, 0, 2, 3, 1};
  spblas::csr_wrapper<T, I> a(3, 5);
  a.update({}, rowptr, colind);

  spblas::operation_state state;
  spblas::sort_matrix(state, a);
  EXPECT_EQ(colind, (std::vector<I>{0, 2, 4, 1, 3}));
}

// Arrays longer than the stored entries: the values still move with the
// indices, and the spare entries are left alone.
TEST(SortMatrix, ArraysLongerThanStoredEntries) {
  using T = float;
  using I = spblas::index_t;

  std::vector<I> rowptr{0, 2};
  std::vector<I> colind{1, 0, 0};
  std::vector<T> values{10, 20};
  spblas::csr_wrapper<T, I> a(values, rowptr, colind, {1, 2}, 2);

  spblas::operation_state state;
  spblas::sort_matrix(state, a);
  EXPECT_EQ(colind, (std::vector<I>{0, 1, 0}));
  EXPECT_EQ(values, (std::vector<T>{20, 10}));
}

TEST(SortMatrix, ArraysTooShortThrow) {
  using T = float;
  using I = spblas::index_t;

  std::vector<I> rowptr{0, 3};
  std::vector<I> colind{2, 1, 0};
  std::vector<T> values{1, 2};
  spblas::csr_wrapper<T, I> a(1, 3);
  a.update(values, rowptr, colind);

  spblas::operation_state state;
  EXPECT_THROW(spblas::sort_matrix(state, a), std::invalid_argument);
  // Nothing was reordered.
  EXPECT_EQ(colind, (std::vector<I>{2, 1, 0}));

  std::vector<I> short_colind{2, 1};
  a.update({}, rowptr, short_colind);
  EXPECT_THROW(spblas::sort_matrix(state, a), std::invalid_argument);
}

TEST(SortMatrix, SpGEMMAndSpADDOutput) {
  using T = double;
  using I = spblas::index_t;

  for (auto&& [m, k, nnz] : util::dims) {
    auto [a_values, a_rowptr, a_colind, a_shape, a_nnz] =
        spblas::generate_csr<T, I>(m, k, nnz, 0);
    auto [b_values, b_rowptr, b_colind, b_shape, b_nnz] =
        spblas::generate_csr<T, I>(k, m, nnz, 1);
    spblas::matrix_opt a(spblas::csr_wrapper<T, I>(a_values, a_rowptr, a_colind,
                                                   a_shape, a_nnz));
    spblas::matrix_opt b(spblas::csr_wrapper<T, I>(b_values, b_rowptr, b_colind,
                                                   b_shape, b_nnz));
    spblas::operation_state sort_state;

    // 2-stage SpGEMM, then sort.
    {
      spblas::csr_wrapper<T, I> c(m, m);
      spblas::operation_state state;
      spblas::multiply_compute(state, T(1), a, b, c);
      std::vector<I> rowptr(m + 1), colind(state.get_result_nnz());
      std::vector<T> values(state.get_result_nnz());
      c.update(values, rowptr, colind);
      spblas::multiply_fill(state, T(1), a, b, c);
      auto before = util::to_dense(c);

      spblas::sort_matrix(sort_state, c);
      EXPECT_TRUE(util::sorted_slices(rowptr, colind));
      EXPECT_EQ(util::to_dense(c), before);
    }

    // 3-stage SpGEMM: sort the pattern once; numeric then fills values into
    // the sorted pattern.
    {
      spblas::csr_wrapper<T, I> c(m, m);
      spblas::operation_state state;
      spblas::multiply_compute_symbolic(state, T(1), a, b, c);
      std::vector<I> rowptr(m + 1), colind(state.get_result_nnz());
      std::vector<T> values;
      c.update(values, rowptr, colind);
      spblas::multiply_fill_symbolic(state, T(1), a, b, c);
      spblas::sort_matrix(sort_state, c);
      EXPECT_TRUE(util::sorted_slices(rowptr, colind));

      values.resize(state.get_result_nnz());
      c.update(values, rowptr, colind);
      spblas::multiply_numeric(state, T(1), a, b, c);
      EXPECT_TRUE(util::sorted_slices(rowptr, colind));

      // Reference: the 2-stage product.
      spblas::csr_wrapper<T, I> c_ref(m, m);
      spblas::operation_state ref_state;
      spblas::multiply_compute(ref_state, T(1), a, b, c_ref);
      std::vector<I> r_rowptr(m + 1), r_colind(ref_state.get_result_nnz());
      std::vector<T> r_values(ref_state.get_result_nnz());
      c_ref.update(r_values, r_rowptr, r_colind);
      spblas::multiply_fill(ref_state, T(1), a, b, c_ref);
      auto expected = util::to_dense(c_ref);
      auto actual = util::to_dense(c);
      for (std::size_t i = 0; i < expected.size(); i++) {
        EXPECT_NEAR(expected[i], actual[i],
                    1e-12 * (1 + std::abs(expected[i])));
      }
    }

    // 2-stage SpADD (A + A, both m x k), then sort.
    {
      spblas::csr_wrapper<T, I> c(m, k);
      spblas::operation_state state;
      spblas::add_compute(state, a, a, c);
      std::vector<I> rowptr(m + 1), colind(state.get_result_nnz());
      std::vector<T> values(state.get_result_nnz());
      c.update(values, rowptr, colind);
      spblas::add_fill(state, a, a, c);
      auto before = util::to_dense(c);

      spblas::sort_matrix(sort_state, c);
      EXPECT_TRUE(util::sorted_slices(rowptr, colind));
      EXPECT_EQ(util::to_dense(c), before);
    }
  }
}
