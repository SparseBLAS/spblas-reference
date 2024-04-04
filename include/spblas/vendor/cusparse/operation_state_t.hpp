#pragma once

#include <cusparse.h>

namespace spblas {

namespace __cusparse {

struct operation_state_t {
  cusparseSpGEMMDescr_t spgemm_descr;
};

} // namespace __cusparse

} // namespace spblas
