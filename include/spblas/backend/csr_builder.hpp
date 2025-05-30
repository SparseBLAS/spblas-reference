#pragma once

#include <spblas/views/csr_view.hpp>

namespace spblas {

namespace __backend {

template <typename T, std::integral I = index_t, std::integral O = I>
class csr_builder {
public:
  csr_builder(csr_view<T, I, O> view) : view_(view) {
    view_.rowptr()[0] = 0;
  }

  template <__ranges::forward_range Row>
  void insert_row(I row_index, Row&& row) {
    if (j_ptr_ + __ranges::size(row) > __ranges::size(view_.values()) ||
        j_ptr_ + __ranges::size(row) > __ranges::size(view_.colind())) {
      throw std::runtime_error("csr_builder: not enough space in CSR.");
    }

    if (row_index + 1 >= __ranges::size(view_.rowptr())) {
      throw std::runtime_error("csr_builder: not enough rows in CSR.");
    }

    while (i_ < row_index) {
      view_.rowptr()[i_ + 1] = j_ptr_;
      i_++;
    }

    for (auto&& [j, v] : row) {
      view_.values()[j_ptr_] = v;
      view_.colind()[j_ptr_] = j;
      j_ptr_++;
    }
    view_.rowptr()[i_ + 1] = j_ptr_;
    i_++;
  }

  void finish() {
    while (i_ < view_.shape()[0]) {
      view_.rowptr()[i_ + 1] = j_ptr_;
      i_++;
    }
  }

private:
  csr_view<T, I, O> view_;
  O j_ptr_ = 0;
  I i_ = 0;
};

template <typename T, std::integral I = index_t, std::integral O = I>
class csc_builder {
public:
  csc_builder(csc_view<T, I, O> view) : builder_(transposed(view)) {}

  template <__ranges::forward_range Column>
  void insert_column(I column_index, Column&& column) {
    builder_.insert_row(column_index, std::forward<Column>(column));
  }

  void finish() {
    builder_.finish();
  }

private:
  csr_builder<T, I> builder_;
};

} // namespace __backend

} // namespace spblas
