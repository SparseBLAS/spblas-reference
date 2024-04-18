#pragma once

#include <span>
#include <spblas/detail/detail.hpp>
#include <spblas/views/view_base.hpp>

#ifdef SPBLAS_ENABLE_CUSPARSE
#include <spblas/vendor/cusparse/types.hpp>
#endif

namespace spblas {

template <typename T, typename I, typename O>
class csr_builder;

template <typename T, std::integral I = index_t, std::integral O = I>
class csr_view : public view_base {
public:
  using scalar_type = T;
  using scalar_reference = T&;
  using index_type = I;
  using offset_type = O;

  csr_view(T* values, O* rowptr, I* colind, index<I> shape, O nnz)
      : values_(values, nnz), rowptr_(rowptr, shape[0] + 1),
        colind_(colind, nnz), shape_(shape), nnz_(nnz) {
    if (rowptr_.data() == nullptr) {
      rowptr_ = std::span<I>((I*) nullptr, (I*) nullptr);
    }
#ifdef SPBLAS_ENABLE_CUSPARSE
    cusparseCreateCsr(&mat_, shape[0], shape[1], values_.size(), rowptr_.data(),
                      colind_.data(), values_.data(),
                      cusparse_index_type<offset_type>(),
                      cusparse_index_type<index_type>(),
                      CUSPARSE_INDEX_BASE_ZERO, cuda_data_type<scalar_type>());
#endif
  }

  template <__ranges::contiguous_range V, __ranges::contiguous_range R,
            __ranges::contiguous_range C>
  csr_view(V&& values, R&& rowptr, C&& colind, index<I> shape, O nnz)
      : values_(__ranges::data(values), __ranges::size(values)),
        rowptr_(__ranges::data(rowptr), __ranges::size(rowptr)),
        colind_(__ranges::data(colind), __ranges::size(colind)), shape_(shape),
        nnz_(nnz) {
#ifdef SPBLAS_ENABLE_CUSPARSE
    cusparseCreateCsr(&mat_, shape[0], shape[1], values_.size(), rowptr_.data(),
                      colind_.data(), values_.data(),
                      cusparse_index_type<offset_type>(),
                      cusparse_index_type<index_type>(),
                      CUSPARSE_INDEX_BASE_ZERO, cuda_data_type<scalar_type>());
#endif
  }

  void update(std::span<T> values, std::span<I> rowptr, std::span<O> colind) {
    values_ = values;
    rowptr_ = rowptr;
    colind_ = colind;
#ifdef SPBLAS_ENABLE_CUSPARSE
    cusparseCsrSetPointers(mat_, rowptr_.data(), colind_.data(),
                           values_.data());
#endif
  }

  void update(std::span<T> values, std::span<I> rowptr, std::span<O> colind,
              index<I> shape, O nnz) {
    values_ = values;
    rowptr_ = rowptr;
    colind_ = colind;
    shape_ = shape;
    nnz_ = nnz;
#ifdef SPBLAS_ENABLE_CUSPARSE
    cusparseDestroySpMat(mat_);
    cusparseCreateCsr(&mat_, shape[0], shape[1], values_.size(), rowptr_.data(),
                      colind_.data(), values_.data(),
                      cusparse_index_type<offset_type>(),
                      cusparse_index_type<index_type>(),
                      CUSPARSE_INDEX_BASE_ZERO, cuda_data_type<scalar_type>());
#endif
  }

  std::span<T> values() const noexcept {
    return values_;
  }
  std::span<I> rowptr() const noexcept {
    return rowptr_;
  }
  std::span<O> colind() const noexcept {
    return colind_;
  }

  index<I> shape() const noexcept {
    return shape_;
  }

  O size() const noexcept {
    return nnz_;
  }

#ifdef SPBLAS_ENABLE_CUSPARSE
  const cusparseSpMatDescr_t& get_descr() const {
    return mat_;
  }

  cusparseSpMatDescr_t& get_descr() {
    return mat_;
  }
#endif

  friend class csr_builder<T, I, O>;

private:
  std::span<T> values_;
  std::span<I> rowptr_;
  std::span<O> colind_;
  index<I> shape_;
  O nnz_;
  cusparseSpMatDescr_t mat_;
};

} // namespace spblas
