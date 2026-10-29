#pragma once

namespace spblas {

// Whether the indices within every row (CSR) or column (CSC) of a matrix's
// stored data are in increasing order, as far as is known.
enum class sort_state { unknown, sorted, unsorted };

// Properties of the data a format wrapper holds (SparseBLAS paper: property
// tags), given at construction or in `update()`:
//
//   csr_wrapper A(values, rowptr, colind, shape, nnz,
//                 property_list{sort_state::sorted});
//
// Users and the library may query and change them at any time (e.g.
// `sort_matrix` sets `sorted`). A property that is set is a claim about the
// stored data, so it must be true; `unknown` claims nothing. Only sortedness
// is supported so far.
class property_list {
public:
  constexpr property_list() = default;

  constexpr property_list(spblas::sort_state s) : sort_state_(s) {}

  constexpr spblas::sort_state sort_state() const noexcept {
    return sort_state_;
  }

  constexpr void set_sort_state(spblas::sort_state s) noexcept {
    sort_state_ = s;
  }

  friend constexpr bool operator==(const property_list&,
                                   const property_list&) = default;

private:
  spblas::sort_state sort_state_ = spblas::sort_state::unknown;
};

} // namespace spblas
