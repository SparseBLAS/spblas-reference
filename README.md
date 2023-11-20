# Sparse BLAS Reference
This repository holds a prototype Sparse BLAS implementation.

## Views
The interface is currently built around views.  Views are non-owning, lightweight objects.
They reference data, but they do not own it.  Views make it easy to use a library without
having to copy data in and out of it.

```cpp
  using namespace spblas;

  // User owns their own data
  int m, n, nnz = ...;
  std::vector<float> values = ...;
  std::vector<int> rowptr = ...;
  std::vector<int> colind = ...;

  // They can create a view
  csr_view<float, int> a(values, rowptr, colind, {m, n}, nnz);

  // We can then perform an operation. No copies required.

  multiply(a, b, c);
```

## Multi-Phase Operations
Multi-phase operations allow implementations to perform inspector-executor style optimizations.
In the inspect phase, an implementation can optimize the format of a matrix or provide other
metadata that will allow the operation to be performed more efficiently.  Having an inspect
phase also allows users to fully control data layout and allocation by having them allocate more
space for the output if needed in a sparse-times-sparse operation.

```cpp
  // Call `inspect` to perform optimization phase.
  // Implementations may apply arbitrary optimizations here.
  auto info = multiply_inspect(a, b, c /*, optimization hints */);

  // If c is sparse, I may need to allocate more memory for the output.
  std::vector<float> c_values(info.result_nnz());
  std::vector<spblas::index_t> c_rowptr(info.result_shape()[0] + 1);
  std::vector<spblas::index_t> c_colind(info.result_nnz());

  c.update(c_values, c_rowptr, c_colind);

  // I can then execute the operation using `info`
  multiply_execute(info, a, b, c);
```
