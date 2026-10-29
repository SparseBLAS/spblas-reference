#pragma once

#include <cstddef>
#include <memory_resource>

#include <spblas/detail/backend_state.hpp>

namespace spblas {

// Where memory from a resource can be accessed.
enum class memory_space { host, device, shared };

// A resource for buffers on the device operations compute on (SparseBLAS
// paper: the allocator "for the memory space where the computations are
// executed"). Like std::pmr::memory_resource, plus the memory space and
// stream-ordered allocation: allocate / deallocate are ordered on the stream
// they are given (the backend's stream type), or are synchronous.
class memory_resource {
public:
  using stream_type = __detail::backend_stream_t;

  virtual ~memory_resource() = default;

  void* allocate(std::size_t bytes, std::size_t alignment, stream_type stream) {
    return do_allocate(bytes, alignment, stream);
  }

  void deallocate(void* p, std::size_t bytes, std::size_t alignment,
                  stream_type stream) {
    do_deallocate(p, bytes, alignment, stream);
  }

  memory_space space() const noexcept {
    return do_space();
  }

private:
  virtual void* do_allocate(std::size_t bytes, std::size_t alignment,
                            stream_type stream) = 0;
  virtual void do_deallocate(void* p, std::size_t bytes, std::size_t alignment,
                             stream_type stream) = 0;
  virtual memory_space do_space() const noexcept = 0;
};

// Non-owning reference to the resource serving an allocator's compute role:
// a spblas::memory_resource, or, on backends that compute in host memory, a
// host std::pmr resource.
class compute_resource_ref {
public:
  using stream_type = __detail::backend_stream_t;

  compute_resource_ref(spblas::memory_resource* resource) noexcept
      : compute_(resource) {}

  compute_resource_ref(std::pmr::memory_resource* resource) noexcept
      : host_(resource) {}

  void* allocate(std::size_t bytes, std::size_t alignment, stream_type stream) {
    return compute_ != nullptr ? compute_->allocate(bytes, alignment, stream)
                               : host_->allocate(bytes, alignment);
  }

  void deallocate(void* p, std::size_t bytes, std::size_t alignment,
                  stream_type stream) {
    if (compute_ != nullptr) {
      compute_->deallocate(p, bytes, alignment, stream);
    } else {
      host_->deallocate(p, bytes, alignment);
    }
  }

  memory_space space() const noexcept {
    return compute_ != nullptr ? compute_->space() : memory_space::host;
  }

  // The resource referred to (one of them is null).
  spblas::memory_resource* compute_resource() const noexcept {
    return compute_;
  }

  std::pmr::memory_resource* host_resource() const noexcept {
    return host_;
  }

private:
  spblas::memory_resource* compute_ = nullptr;
  std::pmr::memory_resource* host_ = nullptr;
};

// Which memory the library uses for an operation_state or matrix_opt
// (SparseBLAS paper: the user-provided allocator). Cheap to copy and
// non-owning: the resources must outlive every object using them.
//
//   host role     library objects (e.g. a matrix_opt's shared data) and host
//                 scratch, as a std::pmr resource. std::pmr containers are
//                 only ever given this role.
//   compute role  buffers on the device operations compute on.
//
// A role left unset (or null) uses the backend's default allocator
// (__detail::default_allocator), for the backend's default execution policy;
// a role the user set is always preferred. On backends that compute in host
// memory (all CPU backends), a host resource given alone serves both roles.
class allocator {
public:
  allocator() noexcept = default;

  allocator(std::nullptr_t) noexcept {}

  allocator(std::pmr::memory_resource* host) noexcept : host_(host) {}

  allocator(const std::pmr::polymorphic_allocator<>& host) noexcept
      : host_(host.resource()) {}

  allocator(spblas::memory_resource* compute,
            std::pmr::memory_resource* host = nullptr) noexcept
      : host_(host), compute_(compute) {}

  // The host role, for library objects and std::pmr scratch.
  std::pmr::polymorphic_allocator<> host() const noexcept;

  // The compute role, for buffers on the compute device.
  compute_resource_ref compute() const noexcept;

private:
  std::pmr::memory_resource* host_ = nullptr;
  spblas::memory_resource* compute_ = nullptr;
};

namespace __detail {

// The backend's default allocator for an execution policy; the policy-less
// overload is for the backend's default policy. Backends that compute in
// host memory use their default memory resource for both roles, whatever
// the policy. A backend whose compute device is not the host must set the
// compute role here.
inline allocator default_allocator() noexcept {
  return allocator(default_memory_resource());
}

template <typename Policy>
allocator default_allocator(const Policy&) noexcept {
  return default_allocator();
}

} // namespace __detail

inline std::pmr::polymorphic_allocator<> allocator::host() const noexcept {
  return host_ != nullptr ? host_ : __detail::default_allocator().host_;
}

inline compute_resource_ref allocator::compute() const noexcept {
  if (compute_ != nullptr) {
    return compute_;
  }
  if constexpr (__detail::backend_computes_on_host) {
    return host().resource();
  } else {
    return __detail::default_allocator().compute_;
  }
}

} // namespace spblas
