#include <gtest/gtest.h>

#include "util.hpp"
#include <spblas/allocator.hpp>
#include <spblas/array.hpp>
#include <spblas/spblas.hpp>

#if defined(SPBLAS_ENABLE_ROCSPARSE)

#include <hip/hip_runtime.h>

template <typename ValueType>
void copy_to_device(std::size_t num, const ValueType* input,
                    ValueType* output) {
  hipMemcpy(output, input, num * sizeof(ValueType), hipMemcpyHostToDevice);
}

template <typename ValueType>
void copy_to_host(std::size_t num, const ValueType* input, ValueType* output) {
  hipMemcpy(output, input, num * sizeof(ValueType), hipMemcpyDeviceToHost);
}

using default_allocator = spblas::detail::rocm_allocator;

#else

#include <algorithm>

template <typename ValueType>
void copy_to_device(std::size_t num, const ValueType* input,
                    ValueType* output) {
  std::copy(input, input + num, output);
}

template <typename ValueType>
void copy_to_host(std::size_t num, const ValueType* input, ValueType* output) {
  std::copy(input, input + num, output);
}

class default_allocator : public spblas::allocator {
  void* alloc(std::size_t size) const override {
    void* ptr = ::operator new(size, std::nothrow_t{});
    return ptr;
  };

  void free(void* ptr) const override {
    ::operator delete(ptr, std::nothrow_t{});
  }
};
#endif

TEST(CsrView, SpMV) {
  using T = float;
  using I = spblas::index_t;
  auto alloc = std::make_shared<const default_allocator>();

  for (auto&& [m, n, nnz] : util::dims) {
    auto [values, rowptr, colind, shape, _] =
        spblas::generate_csr<T, I>(m, n, nnz);
    spblas::detail::array<T> dvalues(alloc, nnz);
    spblas::detail::array<I> drowptr(alloc, m + 1);
    spblas::detail::array<I> dcolind(alloc, nnz);
    copy_to_device(nnz, values.data(), dvalues.get_data());
    copy_to_device(m + 1, rowptr.data(), drowptr.get_data());
    copy_to_device(nnz, colind.data(), dcolind.get_data());
    spblas::csr_view<T, I> a(dvalues.get_data(), drowptr.get_data(),
                             dcolind.get_data(), shape, nnz);
    std::vector<T> b(n, 1);
    std::vector<T> c(m, 0);
    spblas::detail::array<T> db(alloc, n);
    spblas::detail::array<T> dc(alloc, m);
    copy_to_device(n, b.data(), db.get_data());
    copy_to_device(m, c.data(), dc.get_data());
    std::span<T> b_span(db.get_data(), n);
    std::span<T> c_span(dc.get_data(), m);
    std::vector<T> c_ref(m, 0);
    for (I i = 0; i < m; i++) {
      for (I j_ptr = rowptr[i]; j_ptr < rowptr[i + 1]; j_ptr++) {
        I j = colind[j_ptr];
        T v = values[j_ptr];

        c_ref[i] += v * b[j];
      }
    }
    spblas::spmv_state_t spmv_state;

    spblas::multiply(spmv_state, a, b_span, c_span);

    copy_to_host(m, dc.get_const_data(), c.data());
    for (I i = 0; i < c_ref.size(); i++) {
      EXPECT_EQ_(c_ref[i], c[i]);
    }
  }
}

TEST(CsrView, SpMV_Ascaled) {
  using T = float;
  using I = spblas::index_t;
  auto alloc = std::make_shared<const default_allocator>();

  for (auto&& [m, n, nnz] :
       {std::tuple(1000, 100, 100), std::tuple(100, 1000, 10000),
        std::tuple(40, 40, 1000)}) {
    for (auto&& alpha : {-10, 1, 5}) {
      auto [values, rowptr, colind, shape, _] =
          spblas::generate_csr<T, I>(m, n, nnz);
      spblas::detail::array<T> dvalues(alloc, nnz);
      spblas::detail::array<I> drowptr(alloc, m + 1);
      spblas::detail::array<I> dcolind(alloc, nnz);
      copy_to_device(nnz, values.data(), dvalues.get_data());
      copy_to_device(m + 1, rowptr.data(), drowptr.get_data());
      copy_to_device(nnz, colind.data(), dcolind.get_data());
      spblas::csr_view<T, I> a(dvalues.get_data(), drowptr.get_data(),
                               dcolind.get_data(), shape, nnz);
      std::vector<T> b(n, 1);
      std::vector<T> c(m, 0);
      spblas::detail::array<T> db(alloc, n);
      spblas::detail::array<T> dc(alloc, m);
      copy_to_device(n, b.data(), db.get_data());
      copy_to_device(m, c.data(), dc.get_data());
      std::span<T> b_span(db.get_data(), n);
      std::span<T> c_span(dc.get_data(), m);
      std::vector<T> c_ref(m, 0);
      for (I i = 0; i < m; i++) {
        for (I j_ptr = rowptr[i]; j_ptr < rowptr[i + 1]; j_ptr++) {
          I j = colind[j_ptr];
          T v = values[j_ptr];

          c_ref[i] += alpha * v * b[j];
        }
      }
      spblas::spmv_state_t spmv_state;

      spblas::multiply(spmv_state, spblas::scaled(alpha, a), b_span, c_span);

      copy_to_host(m, dc.get_const_data(), c.data());
      for (I i = 0; i < c_ref.size(); i++) {
        EXPECT_EQ_(c_ref[i], c[i]);
      }
    }
  }
}

TEST(CsrView, SpMV_BScaled) {
  using T = float;
  using I = spblas::index_t;
  auto alloc = std::make_shared<const default_allocator>();

  for (auto&& [m, n, nnz] :
       {std::tuple(1000, 100, 100), std::tuple(100, 1000, 10000),
        std::tuple(40, 40, 1000)}) {
    for (auto&& alpha : {-10, 1, 5}) {
      auto [values, rowptr, colind, shape, _] =
          spblas::generate_csr<T, I>(m, n, nnz);
      spblas::detail::array<T> dvalues(alloc, nnz);
      spblas::detail::array<I> drowptr(alloc, m + 1);
      spblas::detail::array<I> dcolind(alloc, nnz);
      copy_to_device(nnz, values.data(), dvalues.get_data());
      copy_to_device(m + 1, rowptr.data(), drowptr.get_data());
      copy_to_device(nnz, colind.data(), dcolind.get_data());
      spblas::csr_view<T, I> a(dvalues.get_data(), drowptr.get_data(),
                               dcolind.get_data(), shape, nnz);
      std::vector<T> b(n, 1);
      std::vector<T> c(m, 0);
      spblas::detail::array<T> db(alloc, n);
      spblas::detail::array<T> dc(alloc, m);
      copy_to_device(n, b.data(), db.get_data());
      copy_to_device(m, c.data(), dc.get_data());
      std::span<T> b_span(db.get_data(), n);
      std::span<T> c_span(dc.get_data(), m);
      std::vector<T> c_ref(m, 0);
      for (I i = 0; i < m; i++) {
        for (I j_ptr = rowptr[i]; j_ptr < rowptr[i + 1]; j_ptr++) {
          I j = colind[j_ptr];
          T v = values[j_ptr];

          c_ref[i] += v * alpha * b[j];
        }
      }
      spblas::spmv_state_t spmv_state;

      spblas::multiply(spmv_state, a, spblas::scaled(alpha, b_span), c_span);

      copy_to_host(m, dc.get_const_data(), c.data());
      for (I i = 0; i < c_ref.size(); i++) {
        EXPECT_EQ_(c_ref[i], c[i]);
      }
    }
  }
}

TEST(CscView, SpMV) {
  using T = float;
  using I = spblas::index_t;

  for (auto&& [m, n, nnz] : util::dims) {
    auto [values, colptr, rowind, shape, _] =
        spblas::generate_csc<T, I>(m, n, nnz);

    spblas::csc_view<T, I> a(values, colptr, rowind, shape, nnz);

    std::vector<T> b(n, 1);
    std::vector<T> c(m, 0);

    spblas::multiply(a, b, c);

    std::vector<T> c_ref(m, 0);

    for (I j = 0; j < n; j++) {
      for (I i_ptr = colptr[j]; i_ptr < colptr[j + 1]; i_ptr++) {
        I i = rowind[i_ptr];
        T v = values[i_ptr];

        c_ref[i] += v * b[j];
      }
    }

    for (I i = 0; i < c_ref.size(); i++) {
      EXPECT_EQ_(c_ref[i], c[i]);
    }
  }
}

TEST(CscView, SpMV_Ascaled) {
  using T = float;
  using I = spblas::index_t;

  for (auto&& [m, n, nnz] : util::dims) {
    for (auto&& alpha : {-10, 1, 5}) {
      auto [values, colptr, rowind, shape, _] =
          spblas::generate_csc<T, I>(m, n, nnz);

      spblas::csc_view<T, I> a(values, colptr, rowind, shape, nnz);

      std::vector<T> b(n, 1);
      std::vector<T> c(m, 0);

      spblas::multiply(spblas::scaled(alpha, a), b, c);

      std::vector<T> c_ref(m, 0);

      for (I j = 0; j < n; j++) {
        for (I i_ptr = colptr[j]; i_ptr < colptr[j + 1]; i_ptr++) {
          I i = rowind[i_ptr];
          T v = values[i_ptr];

          c_ref[i] += alpha * v * b[j];
        }
      }

      for (I i = 0; i < c_ref.size(); i++) {
        EXPECT_EQ_(c_ref[i], c[i]);
      }
    }
  }
}

TEST(CscView, SpMV_Bscaled) {
  using T = float;
  using I = spblas::index_t;

  for (auto&& [m, n, nnz] : util::dims) {
    for (auto&& alpha : {-10, 1, 5}) {
      auto [values, colptr, rowind, shape, _] =
          spblas::generate_csc<T, I>(m, n, nnz);

      spblas::csc_view<T, I> a(values, colptr, rowind, shape, nnz);

      std::vector<T> b(n, 1);
      std::vector<T> c(m, 0);

      spblas::multiply(a, spblas::scaled(alpha, b), c);

      std::vector<T> c_ref(m, 0);

      for (I j = 0; j < n; j++) {
        for (I i_ptr = colptr[j]; i_ptr < colptr[j + 1]; i_ptr++) {
          I i = rowind[i_ptr];
          T v = values[i_ptr];

          c_ref[i] += v * alpha * b[j];
        }
      }

      for (I i = 0; i < c_ref.size(); i++) {
        EXPECT_EQ_(c_ref[i], c[i]);
      }
    }
  }
}
