#pragma once

#include <memory>
#include <typeinfo>
#include <variant>

#include "mkl_wrappers.hpp"

namespace spblas {

namespace __mkl_iespblas {

// A handle of either index width, or none.
using handle_variant = std::variant<std::monostate, sparse_handle<MKL_INT>,
                                    sparse_handle<MKL_INT64>>;

// The handle of width J in `v`, emplacing an empty one if `v` holds none of
// that width.
template <typename J>
sparse_handle<J>& get_or_emplace(handle_variant& v) {
  if (!std::holds_alternative<sparse_handle<J>>(v)) {
    v.template emplace<sparse_handle<J>>();
  }
  return std::get<sparse_handle<J>>(v);
}

// Data kept between the stages of one operation.
struct operation_state_t {
  // SpGEMM / SpADD: the product (or sum) C, created by the compute stage.
  handle_variant c;

  // SpGEMM: the operand handles sp2m's later stages reuse, type-erased, and
  // the type they were stored as.
  std::shared_ptr<void> operands;
  const std::type_info* operands_type = nullptr;

  // SpGEMM: whether C's pattern has been written to the user's arrays.
  bool pattern_filled = false;

  template <typename Operands>
  Operands* get_operands() const noexcept {
    return operands_type != nullptr && *operands_type == typeid(Operands)
               ? static_cast<Operands*>(operands.get())
               : nullptr;
  }

  void reset() noexcept {
    c = std::monostate{};
    operands.reset();
    operands_type = nullptr;
    pattern_filled = false;
  }
};

} // namespace __mkl_iespblas

} // namespace spblas
