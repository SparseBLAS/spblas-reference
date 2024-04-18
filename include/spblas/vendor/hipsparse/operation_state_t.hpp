#pragma once

#include <hipsparse.h>

namespace spblas {

namespace __hipsparse {

struct operation_state_t {
  hipsparseSpGEMMDescr_t spgemm_descr;
};

} // namespace __hipsparse

} // namespace spblas
