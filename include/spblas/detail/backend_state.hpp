#pragma once

#include <memory_resource>

// Per-backend hooks for operation_state and matrix_opt:
//
//   backend_operation_state_t   data shared between the stages of one
//                               operation, held by operation_state
//   backend_matrix_opt_state_t  optimization data for one matrix, held by
//                               matrix_opt; invalidate() is called when the
//                               library changes the matrix's data
//   default_memory_resource()   used when the user provides no allocator
//                               (detail/allocator.hpp: default_allocator)
//   backend_stream_t            what stream-ordered allocations are ordered
//                               on
//   backend_computes_on_host    whether operations compute in host memory,
//                               so a host resource also serves an
//                               allocator's compute role
//
// Backends that have not provided a hook get the reference default.

#ifdef SPBLAS_ENABLE_ONEMKL_SYCL
#include <spblas/vendor/onemkl_sycl/matrix_opt_state_t.hpp>
#endif

namespace spblas {

namespace __detail {

struct empty_operation_state_t {};

// For backends without streams.
struct no_stream {};

// No backend has streams or computes outside host memory yet.
using backend_stream_t = no_stream;
inline constexpr bool backend_computes_on_host = true;

struct empty_matrix_opt_state_t {
  void invalidate() noexcept {}
};

#if defined(SPBLAS_ENABLE_ONEMKL_SYCL)
using backend_operation_state_t = empty_operation_state_t;
using backend_matrix_opt_state_t = __mkl::matrix_opt_state_t;

inline std::pmr::memory_resource* default_memory_resource() noexcept {
  return std::pmr::new_delete_resource();
}
#else
using backend_operation_state_t = empty_operation_state_t;
using backend_matrix_opt_state_t = empty_matrix_opt_state_t;

inline std::pmr::memory_resource* default_memory_resource() noexcept {
  return std::pmr::new_delete_resource();
}
#endif

} // namespace __detail

} // namespace spblas
