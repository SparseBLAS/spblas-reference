#pragma once

class cuda_allocator {
public:
  // we can also put the stream into consturctor to use cudaMallocAsync ...
  void alloc(void** ptrptr, size_t size) {
    cudaMalloc(ptrptr, size);
  }

  void free(void* ptr) {
    cudaFree(ptr);
  }
};
