#pragma once

#include <spblas/views/csr_view.hpp>

namespace spblas {

namespace __backend {

template <typename T, typename I, typename O>
class csr_builder {
public:
  csr_builder(csr_view<T, I, O> view) : view_(view) { view_.rowptr()[0] = 0; }

  template <__ranges::forward_range Row>
  void insert_row(I row_index, Row&& row) {
    if (j_ptr_ + __ranges::size(row) > __ranges::size(view_.values()) ||
        j_ptr_ + __ranges::size(row) > __ranges::size(view_.colind())) {
      throw std::runtime_error("csr_builder: not enough space in CSR.");
    }
    while (i_ < row_index) {
      view_.rowptr()[i_ + 1] = j_ptr_;
    }

    for (auto&& [j, v] : row) {
      view_.values()[j_ptr_] = v;
      view_.colind()[j_ptr_] = j;
      ++j_ptr_;
    }
    view_.rowptr()[i_ + 1] = j_ptr_;
    i_++;
  }

  void complete() {
    while (i_ < view_.shape()[0]) {
      view_.rowptr()[i_ + 1] = j_ptr_;
      i_++;
    }
    view_.nnz_ = j_ptr_;
  }

private:
  csr_view<T, I, O> view_;
  O j_ptr_ = 0;
  I i_ = 0;
};

} // namespace __backend

} // namespace spblas
